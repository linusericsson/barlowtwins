# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time

from PIL import Image, ImageOps, ImageFilter
import numpy as np
from torch import nn, optim
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from comclr_transform import ComCLRTransform

parser = argparse.ArgumentParser(description='ComCLR Training (based on Barlow Twins)')
parser.add_argument('data', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--split', default='train', type=str, metavar='S',
                    help='dataset split to train on (train/val)')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=2048, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--beta', default=1.0, type=float, metavar='L',
                    help='weight of redundancy terms')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')

parser.add_argument('--find-unused-parameters', action='store_true')


def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_JOB_ID' in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
        args.dist_url = f'tcp://{host_name}:58472'
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = 'tcp://localhost:58472'
        args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'comclr_stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = ComCLR(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    if args.find_unused_parameters: print('Warning: find_unused_parameters=True!')
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=args.find_unused_parameters)
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'comclr_checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'comclr_checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0

    data_transform = ComCLRTransform()
    dataset = torchvision.datasets.ImageFolder(args.data / args.split, transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()
    ]))
    if args.split == 'val': print('Warning: loading validation set!')
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler)

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, (x, _) in enumerate(loader, start=epoch * len(loader)):
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                loss = model.forward(x, data_transform, gpu=gpu)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % args.print_freq == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_biases=optimizer.param_groups[1]['lr'],
                                 loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
        if args.rank == 0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / 'comclr_checkpoint.pth')
    if args.rank == 0:
        # save final model
        torch.save(model.module.backbone.state_dict(),
                   args.checkpoint_dir / 'comclr_resnet50.pth')


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def D(a, b): # negative cosine similarity
    return 1 - F.cosine_similarity(a, b, dim=-1).mean()


class ComCLR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        # projectors
        sizes = [2048] + list(map(int, args.projector.split('-')))

        # can't use batch norm in the heads at the moment
        # since my implementation only passes 1 feature at a time through the head
        print('Warning: not using batch norm layers in heads!')
        def make_projector(sizes, batch_norm=True):
            layers = []
            for i in range(len(sizes) - 2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
                if batch_norm: layers.append(nn.BatchNorm1d(sizes[i + 1]))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
            return layers

        self.central_head = nn.Sequential(*make_projector(sizes))
        self.central_bn = nn.BatchNorm1d(sizes[-1], affine=False)

        self.spatial_head = nn.Sequential(*make_projector(sizes))
        self.spatial_bn = nn.BatchNorm1d(sizes[-1], affine=False)
        self.colour_head = nn.Sequential(*make_projector(sizes))
        self.colour_bn = nn.BatchNorm1d(sizes[-1], affine=False)
        self.shape_head = nn.Sequential(*make_projector(sizes))
        self.shape_bn = nn.BatchNorm1d(sizes[-1], affine=False)

        self.heads = {
            'spatial': self.spatial_head, 
            'colour': self.colour_head,
            'shape': self.shape_head,
            'all': self.central_head
        }

    def barlowtwins(self, z1, z2, return_off_diag=True):
        """
        The redundancy term from Barlow Twins
        We skip the on_diag part of the loss here
        which corresponds to our invariance losses
        """
        # empirical cross-correlation matrix
        c = z1.T @ z2

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        if return_off_diag:
            return on_diag + self.args.lambd * off_diag
        else:
            return on_diag

    def forward(self, x, data_transform, gpu):
        # augment images
        y1_central, y2_central = data_transform.augment_batch(x, section='central')
        y1_central, y2_central = y1_central.cuda(gpu, non_blocking=True), y2_central.cuda(gpu, non_blocking=True)
        # pass images through backbone and relevant head
        h1_central, h2_central = self.backbone(y1_central), self.backbone(y2_central)
        z1_central, z2_central = self.central_head(h1_central), self.central_head(h2_central)
        # apply batch normalisation
        z1_central_bn, z2_central_bn = self.central_bn(z1_central), self.central_bn(z2_central)

        # augment images
        y1_spatial, y2_spatial = data_transform.augment_batch(x, section='spatial')
        y1_spatial, y2_spatial = y1_spatial.cuda(gpu, non_blocking=True), y2_spatial.cuda(gpu, non_blocking=True)
        # pass images through backbone and relevant head
        h1_spatial, h2_spatial = self.backbone(y1_spatial), self.backbone(y2_spatial)
        z1_spatial, z2_spatial = self.spatial_head(h1_spatial), self.spatial_head(h2_spatial)
        # apply batch normalisation
        z1_spatial_bn, z2_spatial_bn = self.spatial_bn(z1_spatial), self.spatial_bn(z2_spatial)

        # augment images
        y1_colour, y2_colour = data_transform.augment_batch(x, section='colour')
        y1_colour, y2_colour = y1_colour.cuda(gpu, non_blocking=True), y2_colour.cuda(gpu, non_blocking=True)
        # pass images through backbone and relevant head
        h1_colour, h2_colour = self.backbone(y1_colour), self.backbone(y2_colour)
        z1_colour, z2_colour = self.colour_head(h1_colour), self.colour_head(h2_colour)
        # apply batch normalisation
        z1_colour_bn, z2_colour_bn = self.colour_bn(z1_colour), self.colour_bn(z2_colour)

        # augment images
        y1_shape, y2_shape = data_transform.augment_batch(x, section='shape')
        y1_shape, y2_shape = y1_shape.cuda(gpu, non_blocking=True), y2_shape.cuda(gpu, non_blocking=True)
        # pass images through backbone and relevant head
        h1_shape, h2_shape = self.backbone(y1_shape), self.backbone(y2_shape)
        z1_shape, z2_shape = self.shape_head(h1_shape), self.shape_head(h2_shape)
        # apply batch normalisation
        z1_shape_bn, z2_shape_bn = self.shape_bn(z1_shape), self.shape_bn(z2_shape)

        # compute the losses for each head
        loss = self.barlowtwins(z1_central_bn, z2_central_bn)
        loss += self.barlowtwins(z1_spatial_bn, z2_spatial_bn)
        loss += self.barlowtwins(z1_colour_bn, z2_colour_bn)
        loss += self.barlowtwins(z1_shape_bn, z2_shape_bn)

        # subtract redundancy terms between the central and other heads
        loss -= self.args.beta * (self.barlowtwins(z1_spatial_bn, z1_central_bn, False) + self.barlowtwins(z2_spatial_bn, z2_central_bn, False))
        loss -= self.args.beta * (self.barlowtwins(z1_colour_bn, z1_central_bn, False) + self.barlowtwins(z2_colour_bn, z2_central_bn, False))
        loss -= self.args.beta * (self.barlowtwins(z1_shape_bn, z1_central_bn, False) + self.barlowtwins(z2_shape_bn, z2_central_bn, False))

        return loss


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])



class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


if __name__ == '__main__':
    main()

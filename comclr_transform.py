import torch
from torchvision import transforms

import PIL
import numpy as np
import sys
sys.path.append('../elasticdeform/')
import elasticdeform


# for y1, y2 in loader: # batch size 512, meaning effective batch size of 2048
#    y1_abc, y2_abc = augmentation_for_central(y1, y2) # unconditional augmentations if possible
#    y1_ab, y2_ab = augmentation_for_ab(y1, y2)
#    y1_ac, y2_ac = augmentation_for_ac(y1, y2)
#    y1_bc, y2_bc = augmentation_for_bc(y1, y2)

#    h1, h2 = backbone(y1), backbone(y2)
#    z1_abc, z2_abc = central_head(h1), central_head(h2)
#    z1_ab, z2_ab = spatial_head(h1), spatial_head(h2)
#    z1_ac, z2_ac = colour_head(h1), colour_head(h2)
#    z1_bc, z2_bc = shape_head(h1), shape_head(h2)

#    loss = barlowtwins(z1_abc, z2_abc) # apply central_bn
#    loss += barlowtwins(z1_ab, z2_ab) # apply spatial_bn
#    loss += barlowtwins(z1_ac, z2_ac) # apply colour_bn
#    loss += barlowtwins(z1_bc, z2_bc) # apply shape_bn

#    # redundancy is on_diag of barlowtwins
#    loss += redundancy(z1_ab, z1_abc) + redundancy(z2_ab, z2_abc)
#    loss += redundancy(z1_ac, z1_abc) + redundancy(z2_ac, z2_abc)
#    loss += redundancy(z1_bc, z1_abc) + redundancy(z2_bc, z2_abc)

#    loss.backward()
#    optimizer.step()




class ComCLRTransform(object):
    def __init__(self, resize=256, crop_size=224, norm=[[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]):
        augmentation_dict = {
            'spatial': {
                'aug': transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip()
                ]),
                'type': 'crop'
            },
            'colour/texture': {
                'aug': transforms.Compose([
                    # distort colour
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                    transforms.RandomChoice([
                        transforms.RandomGrayscale(p=1.0),
                        transforms.RandomInvert(p=1.0),
                        transforms.RandomSolarize(threshold=128, p=0.2),
                        # remove texture
                        transforms.RandomApply([transforms.GaussianBlur(11, sigma=(5, 10))], p=1.0),
                        transforms.RandomPosterize(bits=2, p=1.0),
                        # distort texture
                        transforms.RandomAdjustSharpness(sharpness_factor=5, p=1.0),
                        transforms.RandomApply([SaltAndPepperNoise()], p=1.0)
                    ])
                ]),
                'type': 'pil'
            },
            'shape': {
                'aug': transforms.Compose([
                    transforms.RandomApply([Deform()], p=1.0)
                ]),
                'type': 'pre'
            }
        }

        self.augs = [v['aug'] for k, v in augmentation_dict.items()]
        self.types = [v['type'] for k, v in augmentation_dict.items()]

        self.pre_idxs  = [i for i, x in enumerate(self.types) if x == 'pre']
        self.crop_idxs = [i for i, x in enumerate(self.types) if x == 'crop']
        self.pil_idxs  = [i for i, x in enumerate(self.types) if x == 'pil']
        self.post_idxs = [i for i, x in enumerate(self.types) if x == 'post']

        self.pre_transforms = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop_size)
        ])
        self.post_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*norm)
        ])

        self.sections = {
            'central': [1, 1, 1],
            'spatial': [0, 1, 1],
            'colour': [1, 0, 1],
            'shape': [1, 1, 0],
        }

    def sample(self):
        samples = torch.zeros(len(self.augs))
        probs = torch.ones(len(self.augs)) * 0.5
        while sum(samples) == 0:
            samples = torch.bernoulli(probs)
        return samples

    def sample_for_section(self, union):
        union = torch.tensor(union)
        probs = torch.ones(len(self.augs)) * 0.5
        samples_1 = torch.bernoulli(probs) * union
        samples_2 = torch.bernoulli(probs) * union
        while samples_1.sum() == 0:
            samples_1 = torch.bernoulli(probs) * union

        return samples_1, samples_2

    def augment(self, image, samples):
        #print(self.pre_idxs, self.crop_idxs, self.pil_idxs, self.post_idxs)
        #print(samples)
        # first do any 'pre'
        for i in self.pre_idxs:
            if samples[i]:
                image = self.augs[i](image)
                #print(f'Applied {self.augs[i]}')
        # second do either 'crop' or pre_transform
        crop = sum([samples[i] for i in self.crop_idxs]) > 0 # will any crop be applied?
        if crop:
            for i in self.crop_idxs:
                if samples[i]:
                    image = self.augs[i](image)
                    #print(f'Applied {self.augs[i]}')
        else:
            image = self.pre_transforms(image)
            #print(f'Applied {self.pre_transforms}')
        # third do any 'pil'
        for i in self.pil_idxs:
            if samples[i]:
                image = self.augs[i](image)
                #print(f'Applied {self.augs[i]}')
        # fourth do post_transform
        image = self.post_transforms(image)
        #print(f'Applied {self.post_transforms}')
        # fifth do any 'post'
        for i in self.post_idxs:
            if samples[i]:
                image = self.augs[i](image)
                #print(f'Applied {self.augs[i]}')
        return image

    def __call__(self, image, section):
        image = transforms.functional.to_pil_image(image)

        samples_1, samples_2 = self.sample_for_section(self.sections[section])
        image_1 = self.augment(image.copy(), samples_1)
        image_2 = self.augment(image.copy(), samples_2)

        return image_1, image_2#, samples_1, samples_2

    def augment_batch(self, x, section):
        y1s, y2s = [], []
        for _x in x:
            y1, y2 = self.__call__(_x, section)
            y1s.append(y1)
            y2s.append(y2)
        return torch.stack(y1s), torch.stack(y2s)


class SaltAndPepperNoise(object):
    r""" Implements 'Salt-and-Pepper' noise
    Adding grain (salt and pepper) noise
    (https://en.wikipedia.org/wiki/Salt-and-pepper_noise)

    assumption: high values = white, low values = black
    
    Inputs:
            - threshold (float):
            - imgType (str): {"cv2","PIL"}
            - lowerValue (int): value for "pepper"
            - upperValue (int): value for "salt"
            - noiseType (str): {"SnP", "RGB"}
    Output:
            - image ({np.ndarray, PIL.Image}): image with 
                                               noise added
    """
    def __init__(self,
                 threshold_range:float = (0.01, 0.1),
                 imgType:str = "PIL",
                 lowerValue:int = 5,
                 upperValue:int = 250,
                 noiseType:str = "RGB"):
        self.treshold = torch.rand(1).item() * (threshold_range[1] - threshold_range[0]) + threshold_range[0]
        self.imgType = imgType
        self.lowerValue = lowerValue # 255 would be too high
        self.upperValue = upperValue # 0 would be too low
        if (noiseType != "RGB") and (noiseType != "SnP"):
            raise Exception("'noiseType' not of value {'SnP', 'RGB'}")
        else:
            self.noiseType = noiseType
        super(SaltAndPepperNoise).__init__()

    def __call__(self, img):
        if self.imgType == "PIL":
            img = np.array(img)
        if type(img) != np.ndarray:
            raise TypeError("Image is not of type 'np.ndarray'!")
        
        if self.noiseType == "SnP":
            random_matrix = np.random.rand(img.shape[0], img.shape[1])
            img[random_matrix >= (1 - self.treshold)] = self.upperValue
            img[random_matrix <= self.treshold] = self.lowerValue
        elif self.noiseType == "RGB":
            random_matrix = np.random.random(img.shape)      
            img[random_matrix >= (1 - self.treshold)] = self.upperValue
            img[random_matrix <= self.treshold] = self.lowerValue
        
        

        if self.imgType == "cv2":
            return img
        elif self.imgType == "PIL":
            # return as PIL image for torchvision transforms compliance
            return PIL.Image.fromarray(img)


class Deform(object):
    def __init__(self, sigma_range=(10, 50), points_range=(3, 9)):
        self.sigma_range = sigma_range
        self.points_range = points_range
        super(Deform).__init__()

    def __call__(self, img):
        # convert to numpy
        img = np.array(img)
        # apply deformation
        sigma = np.random.rand(1) * (self.sigma_range[1] - self.sigma_range[0]) + self.sigma_range[0]
        points = np.random.randint(low=self.points_range[0], high=self.points_range[1])
        img = elasticdeform.deform_random_grid(img, sigma=sigma, points=points, zoom=1, order=1, mode='nearest', axis=(0, 1))
        # return PIL image
        return PIL.Image.fromarray(img)

from lib import *
from config import *

class ImageTransform:
    def __init__(self,size,mean,std):
        self.data_transform = {
            'train': transforms.Compose(
                [
                    transforms.RandomResizedCrop(size,scale=(0.5,1)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std)
                ]
            ),
            'val': transforms.Compose(
                [
                    transforms.Resize(size),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std)
                ]
            ),
            'test': transforms.Compose(
                [
                    transforms.Resize(size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std)
                ]
            )
        }
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)

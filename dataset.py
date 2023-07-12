from lib import *

class MyDataset:
    def __init__(self, file_list, phase, transform) :
        self.file_list = file_list
        self.phase = phase
        self.transform = transform
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,index):
        img_Path = self.file_list[index]
        image = Image.open(img_Path)

        img_transformed = self.transform(image,self.phase)
        
       
        if 'car' in img_Path:
            label = 0
        elif 'motorbike' in img_Path:
            label = 1
        else:
            label = 2
        return img_transformed, label
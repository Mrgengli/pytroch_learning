import torch
from PIL import Image

class Mydataset(Dataset):
    def __int__(self,txt_path,transform = None):
        super().__int__()
        with open(txt_path,'r') as f:
            for line in f:
                img = line.split()
                imgs.append((img[0],int(img[1])))
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        img,label = self.imgs[index]
        img = Image.open(img).convert('RBG')
        return img, label
    def __len__(self):
         return len(self.imgs)




from torch.utils.data import Dataset
from PIL import Image

class Mydataset(Dataset):
    def __int__(self,txt_path,transform = None,label_transform = None):  #初始化的目的是为了将每一张图片存储到一个列表中
        super().__init__
        imgs = []
        with open(txt_path,'r') as f:
            for line in f:
                line = line.strip()
                words = line.split()
                imgs.append((words[0],int(words[1])))
        self.imgs = imgs

        self.tranform = transform
        self.label_transform = label_transform

    def __getitem__(self,index):
        img_path,label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)




import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image 
import onehot
import torch

class MyData(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.img_path = [os.path.join(self.root_dir,img) for img in os.listdir(root_dir)]
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((60,160)),
                transforms.Grayscale()
            ]
        )

    def __len__(self):
        return self.img_path.__len__()

    def __getitem__(self, index):
        image_path = self.img_path[index]
        image = self.transforms(Image.open(image_path))
        label = self.img_path[index].split("/")[-1].split("_")[0]
        label_tensor = torch.flatten(onehot.text2Vec(label))
        return image,label_tensor

if __name__ == "__main__":
    img,label = MyData("datasets/train")[0]
    print(img.shape,label.shape)
    print(MyData("datasets/train").__len__())
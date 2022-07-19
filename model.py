from torch import nn
import torch

class Model(nn.Module):
    def __init__(self):
        super(Model , self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
        )
        self.layer3 = nn.Sequential(
                nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
        )
        self.layer4 = nn.Sequential(
                nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
        )
        self.layer5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=15360,out_features=4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features=4096,out_features=36*4)
        )
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
if __name__ == "__main__":
    data = torch.ones(40,1,60,160)
    model = Model()
    output = model(data)
    print(data.shape)
    print(torch.argmax(output[0].view(-1,36),1))
    # print(torch.gramax())
from mydataset import MyData
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from model import Model
import torch
from onehot import Vec2text

if __name__ == "__main__":
    train_data = DataLoader(MyData("datasets/train"),batch_size=40,shuffle=True)
    test_data = DataLoader(MyData("datasets/test"),batch_size=1)

    model = Model()
    if torch.cuda.is_available():
        model = model.cuda()

    loss_fn = nn.MultiLabelSoftMarginLoss()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()

    optimizer = optim.Adam(model.parameters(),lr=0.001)

    epoch = 100

    total_train_step = 0

    for i in range(1,epoch+1):
        print("-------------第{}轮训练开始------------".format(i))
        # 训练步骤开始
        model.train()
        for data in train_data:
            imgs,labels = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            outputs = model(imgs)
            loss = loss_fn(outputs,labels)
            
            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_step+=1
            if total_train_step % 5 == 0:
                print("训练次数：{},loss:{}".format(total_train_step,loss.item()))
            
        # 测试步骤
        model.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in test_data:
                imgs,labels = data
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                outputs = model(imgs)
                loss = loss_fn(outputs,labels)
                total_test_loss += loss.item()
                if Vec2text(outputs.view(-1,36)) == Vec2text(labels.view(-1,36)):
                    total_accuracy += 1
            print("整体测试集上的loss:{}".format(total_test_loss))
            print("整体测试集上的正确率:{}".format(total_accuracy/MyData("datasets/test").__len__()))
        if(i%2 == 0):
            torch.save(model,"./models/model_{}.pth".format(i))
            print("模型{}已保存".format(i+1))

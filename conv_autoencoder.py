import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from tqdm import tqdm
import os

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x): # 将vector转成矩阵
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
lr = 1e-3
weight_decay = 1e-5
use_gpu = False # 是否使用gpu优化
download = False # MNIST数据集的下载方式

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

dataset = MNIST(root='.', transform=img_transform, download=download)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            # 前三个参数依次为in_channels 输入图像中的通道数，out_channels 卷积产生的通道数，kernel_size 卷积内核的大小
            # stride 卷积的步幅
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh() # 将输出值映射到[-1, 1]之间
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder()
if use_gpu: model = model.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                             weight_decay=weight_decay)
total_loss = 0
for epoch in range(1, num_epochs + 1):
    processBar = tqdm(dataloader, unit='step') # 构建tqdm进度条
    for img, _ in processBar:
        img = Variable(img) # img是一个b*channel*width*height的矩阵
        if use_gpu: img = img.cuda() # gpu优化

        # forward
        output = model(img) # 前向传播
        loss = criterion(output, img) # 计算损失函数
        # backward
        optimizer.zero_grad() # 清除网络状态（模型的梯度）
        loss.backward() # 反向传播求梯度
        optimizer.step() # 使用迭代器更新模型权重

        processBar.set_description("[{}/{}]".format(epoch, num_epochs))
        
    # ===================log========================
    total_loss += loss.data
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, total_loss))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data) # 将decoder的输出保存成图像
        save_image(pic, './dc_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './conv_autoencoder.pth')
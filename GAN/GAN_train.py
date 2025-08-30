
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset
transform = transforms.Compose([
    transforms.Resize(64), 
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]
])
lr=0.0001
beta1=0.5
beta2=0.999
epochs=30
delta=10
size=64

train_dataset=datasets.ImageFolder(root="D:\\vsc_python\\GAN_project\\data\\data\\train_data",transform=transform)
#专门用来处理内容是图片的数据集
batch_size=64
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
#dcgan对参数初始化很敏感
class Generator(nn.Module):#生成器用dropout可能会增加训练难度

    def __init__(self,size=64):
        super(Generator, self).__init__()
        self.main=nn.Sequential(
            nn.ConvTranspose2d(128, size*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(size*8),
            nn.LeakyReLU(0.2, inplace=True),

            #512*4*4
            nn.ConvTranspose2d(size*8, size*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size*4),
            nn.LeakyReLU(0.2, inplace=True),

            #128*8*8
            nn.ConvTranspose2d(size*4, size*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size*2),
            nn.LeakyReLU(0.2, inplace=True),

            #32*16*16
            nn.ConvTranspose2d(size*2, size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size),
            nn.LeakyReLU(0.2, inplace=True),

            #4*32*32
            nn.ConvTranspose2d(size, 3, 4, 2, 1, bias=False),#因为图像是rgb三通道
            nn.BatchNorm2d(3),
            nn.Tanh(),
            #3*64*64
            
        )
    def forward(self,x):
        return self.main(x)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main=nn.Sequential(
            nn.Conv2d(3, size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size),
            nn.LeakyReLU(0.2, inplace=True),

            #nn.Dropout2d(0.5),#通常用于卷积层
            #16*32*32
            nn.Conv2d(size, size*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size*2),
            nn.LeakyReLU(0.2, inplace=True),

            #nn.Dropout2d(0.5),
            #32*16*16
            nn.Conv2d(size*2, size*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size*4),
            nn.LeakyReLU(0.2, inplace=True),

            #nn.Dropout2d(0.5),
            #64*8*8
            nn.Conv2d(size*4, size*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size*8),
            nn.LeakyReLU(0.2, inplace=True),

            #nn.Dropout2d(0.5),
            #128*4*4
            nn.Conv2d(size*8, 1, 4, 1, 0, bias=False),
            
            nn.Sigmoid(),#最后一层不使用dropout
            #1*1*1
        )
    def forward(self,x):
        return self.main(x).view(-1)#输出一维张量



G=Generator().to(device)
D=Discriminator().to(device)
G.apply(weights_init)
D.apply(weights_init)#apply可将函数应用在所有模块
criterion=nn.BCELoss().to(device)#二元交叉熵
d_optimizer = optim.Adam(D.parameters(), lr=lr/2, betas=(beta1, beta2))
g_optimizer = optim.Adam(G.parameters(), lr=lr*2, betas=(beta1, beta2))

def save_sample_images(epoch, batches_done, generator, fixed_z):
    # 生成并保存图片
    with torch.no_grad():
        generated = generator(fixed_z).cpu()

    # 保存为单独图片文件
    filename = f"epoch{epoch}_batch{batches_done}.png"

    
    
    save_image(generated,filename, normalize=True)
losses = {'d': [], 'g': []}
def train():
    
    total_batches = 0  # 记录总batch数

    for epoch in range(epochs):
        start_time = time.time()  # 每个epoch记录开始时间
        for i, (real_images,_) in enumerate(train_loader):#批量进行
            #必须要加_否则会报错，它可能是某种代表“类别”的东西？
            real_images = real_images.to(device)
            fixed_z = torch.randn(batch_size, 128,1,1, device=device)#用来生成样本的噪声
           

            #batch_size=real_images.size(0)
            # 判别器的损失: 
            real_labels = torch.ones(batch_size, device=device)#由于是批量处理，故bool标签也是向量
            fake_labels=torch.zeros(batch_size,device=device)
            fake_images=G(fixed_z).detach()#detach防止梯度传播到生成器，否则可能会对下面训练生成器产生影响
            d_fake=D(fake_images)#计算假图像的判别器输出，这个过程隐式调用了forward
            d_real=D(real_images)
            d_real_loss=criterion(d_real,real_labels)
            d_fake_loss=criterion(d_fake,fake_labels)
            d_loss=d_real_loss+d_fake_loss#原始loss根据二元交叉熵的近似改写
            
            # 判别器梯度更新: 
            #梯度归零，反向传播，梯度更新
            #利用计算图机制将反向传播和梯度更新联系起来
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            pass


            # 生成器的损失: 
            fake_images=G(fixed_z)#不deatch，保留梯度，因为这相当于正向传播阶段
            d_output=D(fake_images)
            g_loss=criterion(d_output,real_labels)#尽量使判别器的判断为真
            pass
            # 生成器梯度更新: 
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            pass    


            # 记录损失
            losses['d'].append(d_loss.item())
            losses['g'].append(g_loss.item())

            # 可视化过程
            if total_batches % delta== 0 and (epoch+1)%10==0:

                
                

                # 打印进度信息
                print(f"[Epoch {epoch+1}/{epochs}] "
                      f"[Batch {i}/{len(train_loader)}] "
                      f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

            total_batches += 1

        
        if (epoch+1)%10==0:
            z_=torch.randn( 1,128,1,1, device=device)
            save_sample_images(epoch+1, "end", G, z_)
        
            print(f"Epoch {epoch+1} took {time.time() - start_time:.2f} seconds")
    generator_save_path = "dcgan_generator.pth"
    torch.save(G.state_dict(), generator_save_path)  # 只保存权重参数
    print(f"生成器状态字典已保存至: {generator_save_path}")
if __name__=='__main__':
    train()




        

        

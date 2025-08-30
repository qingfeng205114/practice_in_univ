import train
from train import Generator
import torch
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G=Generator().to(device)
#
G.load_state_dict(torch.load("dcgan_generator.pth", map_location=device))
G.eval()

if __name__=='__main__':
    for i in range(10):
        z_=torch.randn( 1,128,1,1, device=device)
        generated=G(z_).detach().cpu()
        filename=f'{i}.png'
        save_image(generated,filename, normalize=True)
    print('done')

import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from torchvision import transforms

def conv_block(in_ch, out_ch, name):
    return nn.Sequential(OrderedDict([
        (name + "conv1", nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)),
        (name + "norm1", nn.BatchNorm2d(out_ch)),
        (name + "relu1", nn.ReLU(inplace=True)),
        (name + "conv2", nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)),
        (name + "norm2", nn.BatchNorm2d(out_ch)),
        (name + "relu2", nn.ReLU(inplace=True)),
    ]))

class TemporalUNet(nn.Module):
    """
    UNet that uses a 'Prior' image as an additional input channel.
    This implements the 'Prior Guessing' algorithm.
    """
    def __init__(self, in_channels=4, out_channels=3, features=16):
        super().__init__()
        self.enc1 = conv_block(in_channels, features, "enc1")
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = conv_block(features, features * 2, "enc2")
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bottleneck = conv_block(features * 2, features * 4, "bottle")
        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(features * 4, features * 2, "dec2")
        self.up1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.dec1 = conv_block(features * 2, features, "dec1")
        self.final = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        return torch.sigmoid(self.final(d1))

def run_inference(cloudy_img, prior_img):
    """
    Combines cloudy image and prior image for cloud shadow removal.
    """
    model = TemporalUNet(in_channels=6, out_channels=3) # RGB + RGB Channels
    model.eval()
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))
    ])
    
    cloudy_tensor = transform(cloudy_img)
    prior_tensor = transform(prior_img)
    
    # Concatenate Cloudy + Prior to form the 6-channel input
    input_tensor = torch.cat([cloudy_tensor, prior_tensor], dim=0).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor).squeeze(0)
    
    return transforms.ToPILImage()(output)

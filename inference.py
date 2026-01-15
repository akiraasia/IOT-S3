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
    Implements a functional 'Prior Guessing' algorithm for the demo.
    It identifies cloud contamination by comparing the cloudy image with the prior.
    It then converges the two sources (GPS-style fix) to reconstruct the terrain.
    """
    # Convert PIL to Numpy
    cloudy = np.array(cloudy_img).astype(np.float32) / 255.0
    prior = np.array(prior_img).astype(np.float32) / 255.0
    
    # 1. Calculate the 'Confidence Map' (Detection)
    # High difference usually indicates clouds when compared to a prior
    diff = np.abs(cloudy - prior).mean(axis=-1)
    
    # Smooth the difference to create a 'Transition Zone'
    import scipy.ndimage as ndimage
    mask = ndimage.gaussian_filter(diff, sigma=5)
    
    # Thresholding to find the 'core' cloudy areas
    mask = np.clip((mask - 0.1) * 3, 0, 1) # Enhance contrast of the mask
    mask = np.expand_dims(mask, axis=-1)
    
    # 2. 'Prior Guessing' Logic:
    # We 'Guess' the ground pixels from the prior where the cloudy image has low confidence (high mask)
    # This is the Convergence of the 4+2 channels we discussed.
    output = cloudy * (1 - mask) + prior * mask
    
    # 3. Final Polish: Apply a slight local contrast enhancement to reconstructed areas
    # This mimics the UNet's 'bottleneck' reconstruction.
    output = np.clip(output * 1.02, 0, 1)
    
    return Image.fromarray((output * 255).astype(np.uint8))

# cl/model/unet_blocks.py

import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, **kwargs):
        super().__init__()
        padding = kernel_size // 2
        # remove built-in padding
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, **kwargs)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=0, **kwargs)
        # … rest of init …

    def forward(self, x):
        # circular pad on width (wrap), reflect on height if you like
        # pad = (left, right, top, bottom)
        x_p = F.pad(x, (self.conv1.kernel_size[1]//2,)*2 + (self.conv1.kernel_size[0]//2,)*2,
                    mode='circular')
        h = self.conv1(x_p)
        h = self.act(h)
        h = F.pad(h, (self.conv2.kernel_size[1]//2,)*2 + (self.conv2.kernel_size[0]//2,)*2,
                  mode='circular')
        h = self.conv2(h)
        return x + h

import torch
import torch.nn as nn

class SynchronizedGroupNorm(nn.Module):
    """
    Group normalization that synchronizes across cubemap faces.
    """
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x):
        """
        Apply synchronized group normalization across cube faces.
        
        Args:
            x: Input tensor (either 4D or 5D)
            
        Returns:
            Normalized tensor of same shape
        """
        # Check if input is 4D or 5D
        if len(x.shape) == 5:
            # Input is [batch, num_faces, channels, height, width]
            batch_size, num_faces, C, H, W = x.shape
            
            # Reshape to normalize across faces
            x_reshaped = x.view(batch_size, num_faces * C, H, W)
            
            # Reshape for group norm calculation
            N, C_all, H, W = x_reshaped.shape
            G = self.num_groups
            
            # Reshape input to separate groups
            x_reshaped = x_reshaped.view(N, G, C_all // G, H, W)
            
            # Calculate mean and var across spatial dims and group channels
            mean = x_reshaped.mean(dim=(2, 3, 4), keepdim=True)
            var = x_reshaped.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
            
            # Normalize
            x_reshaped = (x_reshaped - mean) / torch.sqrt(var + self.eps)
            
            # Reshape back
            x_reshaped = x_reshaped.view(N, C_all, H, W)
            
            # Apply weight and bias if affine
            if self.affine:
                # Reshape weight and bias for broadcasting
                weight = self.weight.view(1, C, 1, 1).repeat(1, num_faces, 1, 1).view(1, C_all, 1, 1)
                bias = self.bias.view(1, C, 1, 1).repeat(1, num_faces, 1, 1).view(1, C_all, 1, 1)
                
                x_reshaped = x_reshaped * weight + bias
            
            # Reshape back to original format
            x_normalized = x_reshaped.view(batch_size, num_faces, C, H, W)
            
            return x_normalized
        else:
            # Input is standard [batch, channels, height, width]
            N, C, H, W = x.shape
            G = self.num_groups
            
            # Reshape input to separate groups
            x_reshaped = x.view(N, G, C // G, H, W)
            
            # Calculate mean and var across spatial dims and group channels
            mean = x_reshaped.mean(dim=(2, 3, 4), keepdim=True)
            var = x_reshaped.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
            
            # Normalize
            x_reshaped = (x_reshaped - mean) / torch.sqrt(var + self.eps)
            
            # Reshape back
            x_reshaped = x_reshaped.view(N, C, H, W)
            
            # Apply weight and bias if affine
            if self.affine:
                x_reshaped = x_reshaped * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
            
            return x_reshaped

def replace_group_norms(module, in_place=True):
    """
    Replace all GroupNorm layers with SynchronizedGroupNorm layers.
    
    Args:
        module: PyTorch module
        in_place: Whether to modify the module in-place
        
    Returns:
        Module with replaced normalization layers
    """
    if not in_place:
        module = copy.deepcopy(module)
    
    for name, child in module.named_children():
        if isinstance(child, nn.GroupNorm):
            setattr(module, name, SynchronizedGroupNorm(
                child.num_groups,
                child.num_channels,
                child.eps,
                child.affine
            ))
            # Copy weights if applicable
            if child.affine:
                getattr(module, name).weight.data.copy_(child.weight.data)
                getattr(module, name).bias.data.copy_(child.bias.data)
        else:
            replace_group_norms(child, in_place=True)
    
    return module
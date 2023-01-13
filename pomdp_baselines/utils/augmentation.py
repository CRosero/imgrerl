
import torch
import torch.nn as nn
import torch.nn.functional as F
from pomdp_baselines.utils import helpers as utl


# .TODO: Same augs as critic_loss when image_augmentation_type!=None?
# .TODO: Make adaptive

from enum import Enum

# class syntax
class AugmentationType(Enum):
    NONE = 1
    SAME_OVER_TIME = 2
    DIFFERENT_OVER_TIME = 3


def augment_observs(observs, image_augmentation_type, single_image_shape):

    with torch.no_grad():
        T_plus_one, B, _ = observs.shape

        if image_augmentation_type==AugmentationType.SAME_OVER_TIME:
            observs = utl.unflatten_images(observs, single_image_shape, normalize_pixels=True, collapse_first_two_dims=False) # ((T+1), B, C, H, W)
            aug = RandomShiftsAug_ConstantOverTime(pad=4)
        elif image_augmentation_type==AugmentationType.DIFFERENT_OVER_TIME:
            observs = utl.unflatten_images(observs, single_image_shape, normalize_pixels=True, collapse_first_two_dims=True) # ((T+1)*B, C, H, W)
            aug = RandomShiftsAug(pad=4)
        
        observs = aug(observs) # ((T+1)*B, C, H, W)
        observs = torch.reshape(observs, (T_plus_one, B, observs.shape[1]*observs.shape[2]*observs.shape[3])) # ((T+1), B, C*H*W)
    
    return observs
        





# Source: https://github.com/facebookresearch/drqv2
class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        img_save_num = 0 # .TODO: Remove
        with torch.no_grad():
            for i in range(0, img_save_num):
                utl.save_image(x[0], "aug", name="1-"+str(i)) # Save image

        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')

        with torch.no_grad():
            for i in range(0, img_save_num):
                utl.save_image(x[0], "aug", name="2-"+str(i)) # Save image

        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1) # repeat grid n times

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        ret = F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False) # .TODO: Use x instead -> avoids copy?
        
        
        with torch.no_grad():
            for i in range(0, img_save_num):
                print(i,": ",(shift[i]*(h + 2 * self.pad)/2))
                utl.save_image(ret[i], "aug", name="3-"+str(i)) # Save image

        return ret



class RandomShiftsAug_ConstantOverTime(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        img_save_num_t, img_save_num_b = 0,0 # .TODO: Remove
        
        t, b, c, h, w = x.size()
        x = x.view(t*b,c,h,w) # reshape cause grid_shape needs 4d
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(t, b, 1, 1, 1)  # Need t*b base grids

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(b, 1, 1, 2), # Only need b different grids
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        shift = shift.repeat(t, 1, 1, 1, 1) # .TODO: does expand also work? repeat copies

        grid = base_grid + shift
        grid = grid.view(t*b,h,w,2) # reshape grid because grid_sample needs 4d and x is in 4d
        
        ret = F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

        with torch.no_grad():
            for i in range(0, img_save_num_t):
                for j in range(0, img_save_num_b):
                    print(f"t:{i}, b:{j} -> {shift[i][j]*(h + 2 * self.pad)/2}")
                    utl.save_image(ret[i*b+j], "newAug", name=f"keepOverTime_t{i}_b{j}") # Save image
        
        return ret
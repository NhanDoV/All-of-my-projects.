import cv2, torch, skimage, os
from torch import nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import torch

class PANDADataset(Dataset):
    def __init__(self, data_dir, mask_dir, df, level = 2, transform=None):
        self.df = df
        self.level = level
        self.transform = transform
        self.data_dir = data_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index, level = 2):
        ID = self.df[index][0]
        data_dir = self.data_dir
        mask_dir = self.mask_dir
        coordinate = self.df[index][1: ]
        image, mask = load_data_and_mask(data_dir, mask_dir, ID, coordinate, level)
        
        return torch.tensor(image).permute(2, 0, 1), torch.tensor(mask).permute(2, 0, 1)[0]
    
def load_data_and_mask(data_dir, mask_dir, ID, coordinates, level = 2):
    """
    Args:
        ID
        coordinates
        level {0, 1, 2}: 
    return : 3D arrays of data & mask image
    """
    data_img = skimage.io.MultiImage(os.path.join(data_dir, f'{ID}.tiff'))[level]
    mask_img = skimage.io.MultiImage(os.path.join(mask_dir, f'{ID}_mask.tiff'))[level]
    coordinates = [coordinate // 2**(2*level) for coordinate in coordinates]
    data_tile = data_img[coordinates[0]: coordinates[1], coordinates[2]: coordinates[3], :]
    mask_tile = mask_img[coordinates[0]: coordinates[1], coordinates[2]: coordinates[3], :]
    data_tile = cv2.resize(data_tile, (512, 512))
    mask_tile = cv2.resize(mask_tile, (512, 512))
    del data_img, mask_img
    
    # Load and return small image
    return data_tile, mask_tile

class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False,
                 batch_norm=False, up_mode='upconv'):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
    
#### Function to make tiles from one image
def split_get_coordinate(img_id, crit = 0.0005, size=512, n_tiles=36):    
    """
    ==================================================================================================
    Input:  img_id (str): image_id from the train dataset, such as '004dd32d9cd167d9cc31c13b704498af'  
            crit (float) in (0, 1): the proportion of the dark_region over whole image (size 256 x 256)
            size (int) : image size
            n_tiles : number of tiles
    return: 
            list of (img_id, x_start, x_end, y_start, y_end) images size 512x512    
            ==========================================================================================
    writen by Nhan
    ==================================================================================================
    """
    img = skimage.io.MultiImage(os.path.join(data_dir, f'{img_id}.tiff'))[0]
    tile_size = 512
    h, w = img.shape[: 2]
    nc = int(w / 512)
    nr = int(h / 512)
    img_ls = []
    coord_ls = []
    S_img_tile = 512*512*3
    
    for i in range(nr):
        for j in range(nc):
            x_start, y_start = int(i*512), int(j*512)
            image_dt = img[ x_start : x_start + 512, y_start : y_start + 512 , :]
            if (image_dt.min() < 185):
                count = len(image_dt[image_dt <= 121])
                if count/(S_img_tile) >= crit:
                    image_dt = cv2.resize(image_dt, (size, size), interpolation = cv2.INTER_AREA)
                    img_ls.append(image_dt)
                    del image_dt, x_start, y_start

    ## choose n_tiles image has a best-view_range 
    img3_dt_ = np.array(img_ls)
    idxs_dt_ = np.argsort(img3_dt_.reshape(img3_dt_.shape[0],-1).sum(-1))[:n_tiles]
    
    ## attach
    list_image = []
    for final_index in idxs_dt_:
        list_image.append(img_ls[final_index])

    yield list_image

#### Funtion to calculate the isup from the list of outputs.
def ISUP(result):
    # result: a list of masks
    # Translation matrix of gleason scores to isup
    import numpy as np
    isup_mat = np.array([[1, 2, 4], [3, 4, 5], [4, 5, 5]])
    # calculate the most dominant gleason score
    p = np.zeros(6)
    for mask in result:
        for i in range(3, 6):
            p[i] += len(np.nonzero((mask == i)*1)) 
    gscore1 = max(np.argmax(p), 3)
    p[gscore1] = 0
    if np.argmax(p) == 0:
        gscore2 = gscore1
    else:
        gscore2 = max(np.argmax(p), 3)
        
    return isup_mat[gscore1 - 3, gscore2 - 3]
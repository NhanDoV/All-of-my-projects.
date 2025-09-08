import os
import numpy as np 
import pandas as pd 
import sys
import gc
import random
import time
from contextlib import contextmanager
from pathlib import Path
from collections import defaultdict, Counter
import skimage.io
import cv2
from PIL import Image
import scipy as sp
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from functools import partial
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import cohen_kappa_score
from collections import OrderedDict
import math
import warnings 
warnings.filterwarnings('ignore')

#=====================================================
class CFG:
    """
        Contains all the model.parameters
    """
    debug = False
    lr = 2e-4
    batch_size = 32
    epochs = 20 # you can train more epochs
    seed = 555
    target_size = 1
    target_col = 'isup_grade'
    n_fold = 6

def init_logger(log_file = 'train.log'):
    """
        For instance: 
            Logs a message with level DEBUG
    """
    from logging import getLogger, DEBUG, FileHandler,  Formatter,  StreamHandler
    log_format = '%(asctime)s %(levelname)s %(message)s'
    
    stream_handler = StreamHandler() 
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(Formatter(log_format))
    
    file_handler = FileHandler(log_file)
    file_handler.setFormatter(Formatter(log_format))
    
    logger = getLogger('PANDA')
    logger.setLevel(DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    
    return logger
#=====================================================
LOG_FILE = 'train.log'
LOGGER = init_logger(LOG_FILE)
#=====================================================
def get_transforms(*, data):
    """
        Normalize both the training & validation datasets then
        converts images/masks (array) to PyTorch Tensors
    """
    assert data in ('train', 'valid')
    
    if data == 'train':
        return Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    elif data == 'valid':
        return Compose([
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')

def seed_torch(seed=42):
    """
        Initialize the random number generator.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def tile(img, sz=128, N=16):
    """
        Split the image into N smaller images then resize this into size (sz x sz)
    """
    shape = img.shape
    # define the padding-values
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz

    #  returns the padded array of rank equal to the given array and the 
    # shape will increase according to pad_width.
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                 constant_values=255)
    
    # resize the image as 5-d array
    img = img.reshape(img.shape[0]//sz, sz, img.shape[1]//sz, sz, 3)

    # transform to 4-dimension as list of (sz, sz, 3)
    img = img.transpose(0,2,1,3,4).reshape(-1, sz, sz, 3)

    if len(img) < N:
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],
                     constant_values=255)

    # sorts the array elements in ascending order and returns indices of the sorted elements
    # Here we only keep the first N elements has significance from the input image
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]

    return img

class TrainDataset(Dataset):
    """
        Process the input image from the training set by
            * spliting 
            * concatenating
            * change color-space
            * augmentated
    """
    def __init__(self, df, labels, transform=None):
        self.df = df
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['image_id'].values[idx]
        # Loading the image.tiff by skimage 
        file_path = f'../input/prostate-cancer-grade-assessment/train_images/{file_name}.tiff'
        image = skimage.io.MultiImage(file_path)[-1]

        # Split the image
        image = tile(image, sz=128, N=16)

        # concat all the images (concatenated by vertical before) by horizontal 
        image = cv2.hconcat([cv2.vconcat([image[0], image[1], image[2], image[3]]), 
                             cv2.vconcat([image[4], image[5], image[6], image[7]]), 
                             cv2.vconcat([image[8], image[9], image[10], image[11]]), 
                             cv2.vconcat([image[12], image[13], image[14], image[15]])])
        
        # Convert (Blue-Green-Red) order to Red-Green-Blue with opencv
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        label = torch.tensor(self.labels[idx]).float()
        
        return image, label    
    
class SEModule(nn.Module):
    """
        Squeeze-and-Excitation Networks with module "forward-propagation"
    """
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
            Here we use these activation functions:
                RELU
                sigmoid
            Beside that we use
                Convolution 2dimension
                Average pooling
            to connect these layers
        """
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
        * Base class for bottlenecks that implements `forward propagation` method.
        * Here we used ReLu as the activation function and linked by 1,2,3 d-convoltution 
        and bn1 (1d - batch normalization) for each layer 
    """
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out

class SEBottleneck(Bottleneck):
    """
        Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

class SEResNetBottleneck(Bottleneck):
    """
        ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
        implementation and uses `stride=stride` in `conv1` and not in `conv2`
        (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

class SEResNeXtBottleneck(Bottleneck):
    """
        ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

class SENet(nn.Module):
    """
        A SENet is a convolutional neural network architecture that employs 
        squeeze-and-excitation blocks to enable the network to perform dynamic 
        channel-wise feature recalibration.

        The main purpose of the Squeeze operation is to extract global information 
        from each of the channels of an image.
    """
    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, 
                            groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 
                                groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

def get_transforms(*, data):
    
    assert data in ('train', 'valid')
    
    if data == 'train':
        return Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    elif data == 'valid':
        return Compose([
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'], \
        'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)
    model.load_state_dict(model.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']


def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext50_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model

def se_resnext101_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext101_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model

class CustomSEResNeXt(nn.Module):
    def __init__(self, model_name='se_resnext50_32x4d'):
        assert model_name in ('se_resnext50_32x4d')
        super().__init__()
        
        self.model = se_resnext50_32x4d(pretrained=None)
        weights_path = pretrained_path[model_name]
        self.model.load_state_dict(torch.load(weights_path))
        self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, 
                                           CFG.target_size)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
def quadratic_weighted_kappa(y_hat, y):
    return cohen_kappa_score(y_hat, y, weights='quadratic')

class OptimizedRounder():
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            else:
                X_p[i] = 5

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5]
        self.coef_ = sp.optimize.minimize(loss_partial, 
                                          initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            else:
                X_p[i] = 5
        return X_p

    def coefficients(self):
        return self.coef_['x']
    
class OptimizedRounder():
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            else:
                X_p[i] = 5

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5]
        self.coef_ = sp.optimize.minimize(loss_partial, 
                                          initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 4
            else:
                X_p[i] = 5
        return X_p

    def coefficients(self):
        return self.coef_['x']
    
def train_fn(folds, device, fold):
    """
        Train your model
    """
    print(f"{20*'=': ^25}{f'N_folds: {fold}': ^10}{20*'=': ^25}")
    optimized_rounder = OptimizedRounder()
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index
    train_dataset = TrainDataset(folds.loc[trn_idx].reset_index(drop=True), 
                                 folds.loc[trn_idx].reset_index(drop=True)[CFG.target_col], 
                                 transform=get_transforms(data='train'))
    valid_dataset = TrainDataset(folds.loc[val_idx].reset_index(drop=True), 
                                 folds.loc[val_idx].reset_index(drop=True)[CFG.target_col], 
                                 transform=get_transforms(data='valid'))
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, 
                              shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, 
                              shuffle=False, num_workers=4)
    
    model = CustomSEResNeXt(model_name='se_resnext50_32x4d')
    model.to(device)
    
    optimizer = Adam(model.parameters(), lr=CFG.lr, amsgrad=False)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, 
                                  patience=2, verbose=True, eps=1e-6)
    
    criterion = nn.MSELoss()
    best_score = -100
    best_loss = np.inf
    best_preds = None
    
    
    for epoch in range(CFG.epochs):
        
        start_time = time.time()
        model.train()
        avg_loss = 0.

        optimizer.zero_grad()
        tk0 = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, (images, labels) in tk0:
            images = images.to(device)
            labels = labels.to(device)            
            y_preds = model(images)
            loss = criterion(y_preds.view(-1), labels)            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            avg_loss += loss.item() / len(train_loader)
            
        model.eval()
        avg_val_loss = 0.
        preds = []
        valid_labels = []
        tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))

        for i, (images, labels) in tk1:            
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                y_preds = model(images)
            
            preds.append(y_preds.to('cpu').numpy())
            valid_labels.append(labels.to('cpu').numpy())

            loss = criterion(y_preds.view(-1), labels)
            avg_val_loss += loss.item() / len(valid_loader)
        
        scheduler.step(avg_val_loss)
            
        preds = np.concatenate(preds)
        valid_labels = np.concatenate(valid_labels)
        
        optimized_rounder.fit(preds, valid_labels)
        coefficients = optimized_rounder.coefficients()
        final_preds = optimized_rounder.predict(preds, coefficients)
        LOGGER.debug(f'Counter preds: {Counter(np.concatenate(final_preds))}')
        LOGGER.debug(f'coefficients: {coefficients}')
        score = quadratic_weighted_kappa(valid_labels, final_preds)

        elapsed = time.time() - start_time
        
        LOGGER.debug(f'  Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.debug(f'  Epoch {epoch+1} - QWK: {score}  coefficients: {coefficients}')
        
        if score > best_score:
            best_score = score
            best_preds = preds
            LOGGER.debug(f'  Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model  coefficients: {coefficients}')
            torch.save(model.state_dict(), f'fold{fold}_se_resnext50.pth')
    return best_preds, valid_labels

class TestDataset(Dataset):
    """
        Process the input image from the validation set by
            * spliting 
            * concatenating
            * change color-space
    """    
    def __init__(self, df, dir_name, transform=None):
        self.df = df
        self.dir_name = dir_name
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['image_id'].values[idx]
        file_path = f'../input/prostate-cancer-grade-assessment/{self.dir_name}/{file_name}.tiff'
        image = skimage.io.MultiImage(file_path)[-1]
        image = tile(image, sz=128, N=16)
        image = cv2.hconcat([cv2.vconcat([image[0], image[1], image[2], image[3]]), 
                             cv2.vconcat([image[4], image[5], image[6], image[7]]), 
                             cv2.vconcat([image[8], image[9], image[10], image[11]]), 
                             cv2.vconcat([image[12], image[13], image[14], image[15]])])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image

def inference(model, test_loader, device):    
    model.to(device)     
    probs = []

    for i, images in enumerate(test_loader):            
        images = images.to(device)
        with torch.no_grad():
            y_preds = model(images)            
        probs.append(y_preds.to('cpu').numpy())
    probs = np.concatenate(probs)
    
    return probs

def submit(device, sample, coefficients, dir_name='test_images'):
    """
        Input params:
            device: using cpu or gpu
            sample: sample_submission.csv
            coefficients : coefficients computed by `OptimizedRounder`
            dir_name : name of directory to data to submit
    """
    if os.path.exists(f'../input/prostate-cancer-grade-assessment/{dir_name}'):
        print(f"{20*'=': ^25}{'run inference': ^16}{20*'=': ^25}")
        test_dataset = TestDataset(sample, dir_name, transform=get_transforms(data='valid'))
        test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False)
        probs = []
        for fold in range(CFG.n_fold):
            model = CustomSEResNeXt(model_name='se_resnext50_32x4d')
            weights_path = f'fold{fold}_se_resnext50.pth'
            model.load_state_dict(torch.load(weights_path, map_location=device))
            _probs = inference(model, test_loader, device)
            probs.append(_probs)
        probs = np.mean(probs, axis=0)
        optimized_rounder = OptimizedRounder()
        preds = optimized_rounder.predict(probs, coefficients)
        sample['isup_grade'] = preds
    return sample
import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from PIL import Image
import json

import torch.optim as optim
from torch.optim import lr_scheduler

from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from sklearn.metrics import f1_score

from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

class PascalImageDataset(Dataset):
    def __init__(self, dataframe, label_list, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.label_list = label_list

    def __len__(self):
        return len(self.dataframe['labels'])

    def __getitem__(self, idx):
        image_filepath = os.path.join(self.img_dir, self.dataframe['fname'][idx])
        image = Image.open(image_filepath).convert("RGB")
        image = transforms.ToTensor()(image)
        image = transforms.Pad(padding=[(500-image.size()[2])//2, (500-image.size()[1])//2])(image)
        image = transforms.ToPILImage()(image)
        image = self.transform(image)

        label = np.zeros(20)
        label_df = pd.DataFrame(self.dataframe['labels'])
        for id, cat in enumerate(self.label_list):
            labels = label_df.iloc[idx].values[0].split(' ')
            for lab in labels:
                if lab == cat:
                    label[id] = 1
        return image, torch.tensor(label)
    
class PascalImageDatasetWithoutAugment(Dataset):
    def __init__(self, dataframe, label_list, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.label_list = label_list

    def __len__(self):
        return len(self.dataframe['labels'])

    def __getitem__(self, idx):
        image_filepath = os.path.join(self.img_dir, self.dataframe['fname'][idx])
        image = Image.open(image_filepath).convert("RGB")
        image = transforms.ToTensor()(image)
        image = transforms.Pad(padding=[(500-image.size()[2])//2, (500-image.size()[1])//2])(image)
        image = transforms.ToPILImage()(image)
        image = self.transform(image)

        label = np.zeros(20)
        label_df = pd.DataFrame(self.dataframe['labels'])
        for id, cat in enumerate(self.label_list):
            labels = label_df.iloc[idx].values[0].split(' ')
            for lab in labels:
                if lab == cat:
                    label[id] = 1
        return image, torch.tensor(label)
    
    
if __name__ == "__main__":
    image_transforms = { 
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'aug': transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    path = 'pascal'

    # Opening JSON file
    f = open(os.path.join(path,'train.json'))

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    # Iterating through the json
    # list
    categories = []
    for i, dic in enumerate(data['categories']):
        categories.append(dic['name'])

    # Closing file
    f.close()
    print(categories)
    
    train_df = pd.read_csv(os.path.join(path,'train.csv'))
    test_df = pd.read_csv(os.path.join(path,'test.csv'))
    # Load the Data

    # Set train and valid directory paths
    train_directory = os.path.join(path, 'train')
    test_directory = os.path.join(path, 'test')

    # Batch size
    batch_size = 128

    # Number of classes
    num_classes = len(categories)
    print(num_classes)

    train_wA = PascalImageDatasetWithoutAugment(dataframe=train_df, label_list=categories, img_dir=os.path.join(path, 'train'), transform=image_transforms['aug'])
    test_wA = PascalImageDatasetWithoutAugment(dataframe=test_df, label_list=categories, img_dir=os.path.join(path, 'test'), transform=image_transforms['test'])

    train = PascalImageDataset(dataframe=train_df, label_list=categories, img_dir=os.path.join(path, 'train'), transform=image_transforms['train'])
    test = PascalImageDataset(dataframe=test_df, label_list=categories, img_dir=os.path.join(path, 'test'), transform=image_transforms['test'])

    # Create iterators for the Data loaded using DataLoader module
    dataloader_wA = {'train': DataLoader(train_wA, batch_size=batch_size, shuffle=True),
                  'test': DataLoader(test_wA, batch_size=batch_size, shuffle=False)}
    dataloader = {'train': DataLoader(train, batch_size=batch_size, shuffle=True),
                  'test': DataLoader(test, batch_size=batch_size, shuffle=False)}
    
    convnext = convnext_large(pretrained=False)
    # Freeze model parameters
    for param in convnext.parameters():
        param.requires_grad = False
        
    layers = [
        nn.Linear(1536, 1000),
        nn.ReLU(),
        nn.BatchNorm1d(1000),
        nn.Dropout(),    
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.BatchNorm1d(500),
        nn.Dropout(),    
        nn.Linear(500, 250),
        nn.ReLU(),
        nn.BatchNorm1d(250),
        nn.Dropout(),    
        nn.Linear(250, 20)
    ]
    convnext.head = nn.Sequential(*layers)
    
    optimizer = optim.Adam(convnext.parameters(), lr=0.001)
    sgdr_partial = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.005)
    
    checkpoint = torch.load("LatestCheckpoint.pt")
    convnext.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    batch_size = checkpoint['batch_size']

    convnext.eval() ## or model.train()
    
    criterion = nn.MultiLabelSoftMarginLoss()

    # specify optimizer
    
    writer = SummaryWriter(log_dir='runs/multi-label')
    
    num_epochs = 20
    
    for epoch in trange(num_epochs, desc="Epochs"):
        result = []
        for phase in ['train', 'test']:
            if phase=='train':
                convnext.train()
            else: convnext.eval()
            
            running_loss = 0.0
            running_corrects = 0.0
            
            if epoch < 3 or epoch == 19:
                for data, target in dataloader_wA[phase]:
                    # data, target = data.to(device), target.to(device)
                    with torch.set_grad_enabled(phase=='train'):
                        output = convnext(data)
                        loss = criterion(output, target)
                        preds = torch.sigmoid(output).data > 0.5
                        preds = preds.to(torch.float32)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            sgdr_partial.step()
                            optimizer.zero_grad()
                    running_loss += loss.item() * data.size(0)
                    running_corrects += f1_score(target.to(torch.int).numpy() ,preds.to(torch.int).numpy() , average="samples")  * data.size(0)

                epoch_loss = running_loss / len(dataloader_wA[phase].dataset)
                epoch_acc = running_corrects / len(dataloader_wA[phase].dataset)

                result.append('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                writer.add_scalar('{} loss:'.format(phase), epoch_loss, global_step = epoch + 1)
                writer.add_scalar('{} acc:'.format(phase), epoch_acc, global_step = epoch + 1)
            else:
                for data, target in dataloader[phase]:
                    # data, target = data.to(device), target.to(device)
                    with torch.set_grad_enabled(phase=='train'):
                        output = convnext(data)
                        loss = criterion(output, target)
                        preds = torch.sigmoid(output).data > 0.5
                        preds = preds.to(torch.float32)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            sgdr_partial.step()
                            optimizer.zero_grad()
                    running_loss += loss.item() * data.size(0)
                    running_corrects += f1_score(target.to(torch.int).numpy() ,preds.to(torch.int).numpy() , average="samples")  * data.size(0)

                epoch_loss = running_loss / len(dataloader[phase].dataset)
                epoch_acc = running_corrects / len(dataloader[phase].dataset)

                result.append('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                writer.add_scalar('{} loss:'.format(phase), epoch_loss, global_step = epoch + 1)
                writer.add_scalar('{} acc:'.format(phase), epoch_acc, global_step = epoch + 1)
        print(result)
    writer.close()
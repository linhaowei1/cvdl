import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import math
from torch.nn import DataParallel
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
import json
import torch
import torch.nn as nn
from skimage import io, color
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import EfficientNet
from torch.utils.data import random_split
from tqdm import tqdm
import pandas as pd
import pdb
from tensorboardX import SummaryWriter
from PIL import ImageFile
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True
DATA_PATH = '/home/linhw/myproject/data_scene'

import argparse
from tqdm import tqdm
from loguru import logger

class LabelSmoothCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, label, smoothing=0.1):
        pred = F.softmax(pred, dim=1)
        one_hot_label = F.one_hot(label, pred.size(1)).float()
        smoothed_one_hot_label = (
            1.0 - smoothing) * one_hot_label + smoothing / pred.size(1)
        loss = (-torch.log(pred)) * smoothed_one_hot_label
        loss = loss.sum(axis=1, keepdim=False)
        loss = loss.mean()

        return loss

class SceneDataset(Dataset):
    def __init__(self, annotations_csv, root_dir, transform=None):
        self.annotations = pd.read_csv(annotations_csv)
        self.root_dir = root_dir
        self.transform = transform
                    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        label = torch.tensor(int(self.annotations.iloc[index, 1]))
        if self.transform:
            image = self.transform(image)
        return [image, label]

def lr_schedule_func(epoch):
    if epoch < 150:
        return 0.1
    elif epoch < 250:
        return 0.01
    else:
        return 0.001 

def train(model, optimizer, dataloader, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    criterion = LabelSmoothCELoss().cuda()
    for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader), desc="training"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_acc = correct/total
    running_loss = train_loss/(batch_idx+1)

    return train_acc, running_loss

def test(model, dataloader, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_num = 0 
    criterion = nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader), desc="testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_num += 1 
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = correct/total
    running_loss = test_loss/len(dataloader)
    return test_acc,running_loss


def main():
    parser = argparse.ArgumentParser(description='PyTorch Imagenette Training')
    parser.add_argument('--device', default='cuda:0',
                        type=str, required=False, help='GPU ids')
    parser.add_argument('--epoch', default=350,
                        type=int, required=True, help='training epochs')
    parser.add_argument('--alldata', dest='alldata', action='store_true',
                        help='use alldata to train')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--weight_decay', '-w', default=5e-4, type=float, help='weight_decay')
    parser.add_argument('--batch_size', default=128, type=int, help='training batch size')
    parser.add_argument('--output_dir', default='./checkpoint' ,type=str)
    parser.add_argument('--warm_up_epochs', default=10, type=int)
    parser.add_argument('--log_file', type=str, default='./log/default.log')
    parser.add_argument('--params', type=str, default=None)
    args = parser.parse_args()

    log_file_name = args.log_file
    logger.add(log_file_name)
    logger.info('args:\n' + args.__repr__())

    batch_size = args.batch_size
    output_dir = args.output_dir 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = args.device
    best_acc = 0  # best test accuracy

    # Data
    logger.info('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    train_dir = os.path.join(DATA_PATH, 'train/train')
    if args.alldata == False:
        whole_set = SceneDataset(annotations_csv='/home/linhw/myproject/data_scene/train_labels.csv',
                                root_dir=train_dir,transform=transform_train)                

        whole_set2 = SceneDataset(annotations_csv='/home/linhw/myproject/data_scene/train_labels.csv',
                                root_dir=train_dir,transform=transform_test)      
        whole_len = len(whole_set)
        train_len = int(whole_len * 0.9)
        val_len = whole_len - train_len
        indices = random.sample(range(0, whole_len), train_len)
        indices2 = []
        for x in range(0,whole_len):
            if x not in indices:
                indices2.append(x)
        
        trainset = torch.utils.data.Subset(whole_set, indices)
        testset = torch.utils.data.Subset(whole_set2, indices2)
    else:
        trainset = SceneDataset(annotations_csv='/home/linhw/myproject/data_scene/train_labels.csv',
                                root_dir=train_dir,transform=transform_train) 
        testset = torch.utils.data.Subset(
                        SceneDataset(annotations_csv='/home/linhw/myproject/data_scene/train_labels.csv',
                                root_dir=train_dir,transform=transform_test),
                        range(8000))

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=8)

    # Model
    logger.info('==> Building model..')
    model = EfficientNet.from_name('efficientnet-b0', num_classes=100)
    model = model.to(device)
    if args.params is not None:
        model.load_state_dict(torch.load(args.params, map_location=device))


    #optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=0.9, weight_decay=args.weight_decay)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warm_up_with_cosine_lr = lambda epoch: epoch / args.warm_up_epochs if epoch <= args.warm_up_epochs else 0.1 * (math.cos((epoch - args.warm_up_epochs) /(args.epoch - args.warm_up_epochs) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    best_acc = 0
    for epoch in range(args.epoch):
        logger.info("Epoch {} started".format(epoch))

        train_acc,training_loss = train(model, optimizer, trainloader, device)
        logger.info("train acc = {:.4f}, training loss = {:.4f} lr = {:.4f}".format(train_acc, training_loss, warm_up_with_cosine_lr(epoch)))
        
        test_acc, test_loss = test(model, testloader, device)
        logger.info("test acc = {:.4f}, test loss = {:.4f}".format(test_acc, test_loss))

        if test_acc > best_acc:
            best_acc = test_acc
            logger.info("best acc improved to {:.4f}".format(best_acc))
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), '{}/bset_model.pt'.format(output_dir))
            logger.info("model saved to {}/bset_model.pt".format(output_dir))
        
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), '{}/last_model.pt'.format(output_dir))
        logger.info("model saved to {}/last_model.pt".format(output_dir))
        
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), '{}/model.pt'.format(output_dir))
        logger.info("model saved to {}/model.pt".format(output_dir))
        scheduler.step()

        logger.info("Epoch {} ended, best acc = {:.4f}".format(epoch, best_acc))

    logger.info("Training finished, best_acc = {:.4f}".format(best_acc))
    
        

if __name__ == '__main__':
    main()

import os
import json
import torch
import torch.nn as nn
from skimage import io, color
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import EfficientNet
from torch.utils.data import random_split
from tqdm import tqdm
import pandas as pd
import pdb
from tensorboardX import SummaryWriter
import random
from model import resnet18, resnet34, resnet50, resnet101, resnet152, wide_resnet101_2, wide_resnet50_2, resnext101_32x8d, resnext50_32x4d, resnet152
from utils import get_args
from PIL import ImageFile
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True
DATA_PATH = '/home/linhw/myproject/data_scene'

def lr_schedule_func(epoch):
    if epoch < 150:
        return 0.001
    else:
        return 0.0001 

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

def train():
    bestacc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        train_bar = tqdm(train_loader)
        for batch, data in enumerate(train_bar):
            images, labels = data
            logits = model(images.to(cuda_device))
            optimizer.zero_grad()
            loss = loss_func(logits, labels.to(cuda_device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.5f}".format(epoch+1, args.epochs, total_loss / (len(train_bar) * (batch + 1)))

        writer.add_scalar('train/loss', total_loss / train_num, epoch)
        
        if args.mode == 'adjust_parameter':
            model.eval()
            acc = 0.0
            with torch.no_grad():
                val_bar = tqdm(val_loader)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outp = model(val_images.to(cuda_device))
                    pred = torch.max(outp, dim=1)[1]
                    acc += torch.eq(pred, val_labels.to(cuda_device)).sum().item()

                    val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, args.epochs)
                
            acc /= val_num
            if acc > bestacc:
                print('save model.')
                bestacc = acc
                torch.save(model.state_dict(), 'best_'+args.save_path)
            torch.save(model.state_dict(), 'last_'+args.save_path)
            print('save last model.')
            writer.add_scalar('val/acc', acc, epoch)

        print('[epoch %d] train_loss: %.5f  val_accuracy: %.5f ' %
        (epoch + 1, total_loss / train_num, acc))
        scheduler.step()

    print("Finished Training!")
    torch.save(model.state_dict(), args.save_path)

def test():
    model.eval()
    acc = 0.0
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for test_data in test_bar:
            test_images, test_labels = test_data
            outp = model(test_images.to(cuda_device))
            pred = torch.max(outp, dim=1)[1]
            acc += torch.eq(pred, test_labels.to(cuda_device)).sum().item()

        acc /= test_num
        print("test acc = {:.5f}".format(acc))


if __name__ == '__main__':
    args = get_args()
    writer = SummaryWriter(log_dir=args.log_dir)
    cuda_device = args.cuda
    with torch.cuda.device(int(cuda_device[-1])):
        # 设置随机种子
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)

        if args.model == 'resnet34':
            model = resnet34(pretrained=args.pretrain).cuda()
        elif args.model == 'resnet18':
            model = resnet18(pretrained=args.pretrain).cuda()
        elif args.model == 'resnet50':
            model = resnet50(pretrained=args.pretrain).cuda()
            print('use model', args.model)
        elif args.model == 'resnet101':
            model = resnet101(pretrained=args.pretrain).cuda()
        elif args.model == 'resnet152':
            model = resnet152(pretrained=args.pretrain).cuda()
        elif args.model == 'resnext50_32x4d':
            model = resnext50_32x4d(pretrained=args.pretrain).cuda()
        elif args.model == 'resnext101_32x8d':
            model = resnext101_32x8d(pretrained=args.pretrain).cuda()
        elif args.model == 'wide_resnet50_2':
            model = wide_resnet50_2(pretrained=args.pretrain).cuda()
        elif args.model == 'wide_resnet101_2':
            model = wide_resnet101_2(pretrained=args.pretrain).cuda()
        elif args.model == 'efficientnet_b3':
            model = EfficientNet.from_name('efficientnet-b3', num_classes=100).cuda()
        elif 'efficientnet' in args.model:
            model = EfficientNet.from_name(args.model, num_classes=100).cuda()
        # 修改channel数（1000分类->100分类）
        if 'resnet' in args.model:
            in_channel = model.fc.in_features
            model.fc = nn.Linear(in_channel, 100)
        
        model.to(cuda_device)
        
        data_transform = {
            "train": transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                    transforms.RandomGrayscale(p=0.1),
                                    transforms.ToTensor()]),
            "test": transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])}

        if args.mode == 'adjust_parameter':
            # 建立损失函数
            if args.model_params_path is not None:
                model_params_path = args.model_params_path
                assert os.path.exists(model_params_path), "file {} does not exist.".format(model_params_path)
                model.load_state_dict(torch.load(model_params_path, map_location=cuda_device))
            
            loss_func = nn.CrossEntropyLoss()
            loss_func.to(cuda_device)
        
            # 建立优化器
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.Adam(params, lr=args.lr, eps=args.eps, weight_decay=0.0001)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lr_schedule_func )

            train_dir = os.path.join(DATA_PATH, 'train/train')
            whole_set = SceneDataset(annotations_csv='/home/linhw/myproject/data_scene/train_labels.csv',
                                    root_dir=train_dir,transform=data_transform['train']) 
            whole_set2 = SceneDataset(annotations_csv='/home/linhw/myproject/data_scene/train_labels.csv',
                                        root_dir=train_dir,transform=data_transform['test'])               
            if args.alldata == False:
                whole_len = len(whole_set)
                train_len = int(whole_len * 0.9)
                val_len = whole_len - train_len
                indices = random.sample(range(0, whole_len), train_len)
                indices2 = []
                for x in range(0,whole_len):
                    if x not in indices:
                        indices2.append(x)
                
                train_set = torch.utils.data.Subset(whole_set, indices)
                validate_set = torch.utils.data.Subset(whole_set2, indices2)
            else:
                train_set = whole_set
                validate_set = whole_set2
            number_worker = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
            print('Using {} dataloader workers every process'.format(number_worker))

            train_loader = torch.utils.data.DataLoader(train_set,
                                                batch_size=args.batch_size, shuffle=True,
                                                num_workers=number_worker)
            val_loader = torch.utils.data.DataLoader(validate_set,
                                                batch_size=args.batch_size, shuffle=True,
                                                num_workers=number_worker)
            train_num = len(train_set)
            val_num = len(validate_set)
            print("using {} images for training, {} images for validation.".format(train_num, val_num))
            #pdb.set_trace()
            train()

        if args.mode == 'train':
            # 建立损失函数
            loss_func = nn.CrossEntropyLoss()
            loss_func.to(cuda_device)
        
            # 建立优化器
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.Adam(params, lr=args.lr, eps=args.eps)
            train_set = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                            transform=data_transform["train"])
            
            number_worker = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
            print('Using {} dataloader workers every process'.format(number_worker))

            train_loader = torch.utils.data.DataLoader(train_set,
                                                batch_size=args.batch_size, shuffle=True,
                                                num_workers=number_worker)
            
            train_num = len(train_set)
            print("using {} images for training.".format(train_num))
            #pdb.set_trace()
            train()

        if args.mode == 'predict':
            data_transform['test'] = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
            DATA_PATH = '/home/linhw/myproject/cvdl_data'
            model_params_path = args.model_params_path
            assert os.path.exists(model_params_path), "file {} does not exist.".format(model_params_path)
            model.load_state_dict(torch.load(model_params_path, map_location=cuda_device))
        
            test_dir = os.path.join(DATA_PATH, 'test')
            test_set = datasets.ImageFolder(test_dir, transform=data_transform['test'])
            #pdb.set_trace()
            number_worker = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
            print('Using {} dataloader workers every process'.format(number_worker))

            test_loader = torch.utils.data.DataLoader(test_set,
                                                batch_size=args.batch_size, shuffle=False,
                                                num_workers=number_worker)
            test_num = len(test_set)
            print("using {} images for predicting.".format(test_num))
            model.eval()
            acc = 0.0
            lst = []
            idx = []
            with torch.no_grad():
                test_bar = tqdm(test_loader)
                for batch, test_data in enumerate(test_bar):
                    test_images, test_labels = test_data
                    outp = model(test_images.to(cuda_device))
                    pred = torch.max(outp, dim=1)[1]
                    lst += pred.detach().cpu().numpy().tolist()
                    idx += test_labels.detach().cpu().numpy().tolist()
                    acc += torch.eq(pred, test_labels.to(cuda_device)).sum().item()
            dat = pd.DataFrame({'Id': ['0007'+ '0'*(4-len(str(i)))+ str(i)+'.jpg' for i in range(len(test_set))], 'Category': lst})
            dat.to_csv(args.csv, index=False)
            acc /= test_num
            print("test acc = {:.5f}".format(acc))

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
        elif 'efficientnet' in args.model:
            model = EfficientNet.from_name(args.model, num_classes=100).cuda()
        else:
            raise NotImplementedError
        # 修改channel数（1000分类->100分类）
        if 'resnet' in args.model:
            in_channel = model.fc.in_features
            model.fc = nn.Linear(in_channel, 100)
        
        model.to(cuda_device)
        
        data_transform = dict()

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
        else:
            raise NotImplementedError

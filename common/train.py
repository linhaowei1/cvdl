from utils.utils import load_checkpoint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from datasets import get_dataset
from common.common import parse_args
from utils.utils import load_checkpoint

P = parse_args()
if torch.cuda.is_available():
    torch.cuda.set_device(0)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
train_set, test_set = get_dataset(P, test_only=False)

kwargs = {'pin_memory':False, 'num_workers': 8}
train_loader = DataLoader(train_set, shuffle=True, batch_size=P.batch_size, **kwargs)
test_loader = DataLoader(test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)

## initialize model ##

if P.model == 'EfficientNet':
    if P.pretrain_params is not None:
        model = EfficientNet.from_pretrained(P.pretrain_params, num_classes=10).to(device)
    else:
        model = EfficientNet(num_classes=10).to(device)
else:
    raise NotImplementedError

criterion = nn.CrossEntropyLoss().to(device)

if P.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
    lr_decay_gamma = 0.1
elif P.optimizer == 'lars':
    from torchlars import LARS
    base_optimizer = optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
    optimizer = LARS(base_optimizer, eps=1e-8, trust_coef=0.001)
    lr_decay_gamma = 0.1
else:
    raise NotImplementedError()

if P.lr_scheduler == 'cosine':
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, P.epochs)
elif P.lr_scheduler == 'step_decay':
    milestones = [int(0.5 * P.epochs), int(0.75 * P.epochs)]
    scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=lr_decay_gamma, milestones=milestones)
else:
    raise NotImplementedError()

from training.scheduler import GradualWarmupScheduler
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=10.0, total_epoch=P.warmup, after_scheduler=scheduler)

if P.resume_path is not None:
    resume = True
    model_state, optim_state, config = load_checkpoint(P.resume_path, mode='last')
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optim_state)
    start_epoch = config['epoch']
    best = config['best']
    error = 100.0
else:
    resume = False
    start_epoch = 1
    best = 100.0
    error = 100.0

if P.load_path is not None:
    checkpoint = torch.load(P.load_path)
    model.load_state_dict(checkpoint)


import time

import torch.optim
import torch.optim.lr_scheduler as lr_scheduler

from utils.utils import AverageMeter, normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup(P):

    def train(P, epoch, model, criterion, optimizer, scheduler, loader, logger=None):
        if epoch == 1:
            # define optimizer and save in P (argument)
            milestones = [int(0.6 * P.epochs), int(0.75 * P.epochs), int(0.9 * P.epochs)]

            optim = torch.optim.SGD(linear.parameters(),
                                        lr=1e-1, weight_decay=P.weight_decay)
            P.optim = optim
            P.scheduler = lr_scheduler.MultiStepLR(P.optim, gamma=0.1, milestones=milestones)
        
        if logger is None:
            log_ = print
        else:
            log_ = logger.log
        
        data_time = AverageMeter()
        batch_time = AverageMeter()
        loss = AverageMeter()
        check = time.time()

        for n, (images, labels) in enumerate(loader):
            model.eval()
            data_time.update(time.time() - check)
            check = time.time()
            batch_size = images[0].size(0)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss_ce = criterion(outputs, labels)

            P.optim.zero_grad()
            loss_ce.backward()
            P.optim.step()

            lr = P.optim.param_groups[0]['lr']

            batch_time.update(time.time() - check)

            loss.update(loss_ce.item(), batch_size)

            if n % 50 == 0:
                log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                    '[LossC %f]' %
                    (epoch, n, batch_time.value, data_time.value, lr,
                    loss.value, ))
            
        P.scheduler.step()

        log_('[DONE] [Time %.3f] [Data %.3f] [LossC %f]' %
            (batch_time.average, data_time.average,
            loss.average))

        if logger is not None:
            logger.scalar_summary('train/loss_ce', loss.average, epoch)
            logger.scalar_summary('train/batch_time', batch_time.average, epoch)
    
    if P.suffix is not None:
        fname += f'_{P.suffix}'
        
    return train, fname








import time
import itertools
import pdb
from sklearn.metrics import f1_score
import diffdist.functional as distops
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from utils.utils import AverageMeter, set_random_seed, normalize

def error_k(output, target, ks=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(ks)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in ks:
        correct_k = correct[:k].view(-1).float().sum(0)
        results.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return results

@torch.no_grad
def test_classifier(P, model, loader, steps, logger=None):
    error = AverageMeter()
    if logger is None:
        log_ = print
    else:
        log_ = logger.log
    
    model.eval() 

    for n, (images, labels) in enumerate(loader):
        batch_size = images.size(0)

        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        top1, = error_k(outputs.data, labels, ks=(1,))
        error.update(top1.item(), batch_size)

        if n % 100 == 0:
            log_('[Test %3d] [TestErr %.3f]' %
                 (n, error.value))
    
    log_(' * [Error@1 %.3f]' % error.average )

    if logger is not None:
        logger.scalar_summary('eval/clean_error', error.average, steps)

    model.train()

    return error.average

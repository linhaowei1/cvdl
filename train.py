from utils.utils import Logger
from utils.utils import save_checkpoint
from utils.utils import save_linear_checkpoint

from common.train import *
from evals import test_classifier
import os

from training import setup
train, fname = setup(P)

logger = Logger(fname, ask=not resume)
logger.log(P)
logger.log(model)

for epoch in range(start_epoch, P.epochs + 1):
    logger.log_dirname(f"Epoch {epoch}")
    model.train()

    train(P, epoch, model, criterion, optimizer, scheduler_warmup, train_loader, logger=logger)
    
    if epoch % P.save_step == 0:
        save_states = model.state_dict()
        save_checkpoint(epoch, save_states, optimizer.state_dict(), logger.logdir)
        save_linear_checkpoint(optim.state_dict(), logger.logdir)
        
        error = test_classifier(P, model, test_loader, epoch, logger=logger)

        is_best = (best > error)
        if is_best:
            best = error

        logger.scalar_summary('eval/best_error', best, epoch)
        logger.log('[Epoch %3d] [Test %5.2f] [Best %5.2f]' % (epoch, error, best))



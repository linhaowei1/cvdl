from argparse import ArgumentParser


def parse_args(default=False):
    """Command-line argument parser for training."""

    parser = ArgumentParser(description='Codes for scene classification')

    parser.add_argument('--model', help='Model', default='EfficientNet',
                        choices=['EfficientNet'], type=str)
    parser.add_argument('--pretrain_params', help='pretrain params', 
                        default=None,type=str)
    parser.add_argument('--resume_path', help='Path to the resume checkpoint',
                        default=None, type=str)
    parser.add_argument('--val_size', help='example numbers for validation set',
                        default=10, type=int)
    parser.add_argument('--load_path', help='Path to the loading checkpoint',
                        default=None, type=str)
    parser.add_argument('--suffix', help='Suffix for the log dir',
                        default=None, type=str)
    parser.add_argument('--save_step', help='Epoch steps to save models',
                        default=10, type=int)

    ##### Training Configurations #####
    parser.add_argument('--epochs', help='Epochs',
                        default=1000, type=int)
    parser.add_argument('--optimizer', help='Optimizer',
                        choices=['sgd', 'lars'],
                        default='lars', type=str)
    parser.add_argument('--lr_scheduler', help='Learning rate scheduler',
                        choices=['step_decay', 'cosine'],
                        default='cosine', type=str)
    parser.add_argument('--warmup', help='Warm-up epochs',
                        default=10, type=int)
    parser.add_argument('--lr_init', help='Initial learning rate',
                        default=1e-1, type=float)
    parser.add_argument('--weight_decay', help='Weight decay',
                        default=1e-6, type=float)
    parser.add_argument('--batch_size', help='Batch size',
                        default=128, type=int)
    parser.add_argument('--test_batch_size', help='Batch size for test loader',
                        default=128, type=int)

    if default:
        return parser.parse_args('')  # empty string
    else:
        return parser.parse_args()

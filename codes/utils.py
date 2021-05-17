import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--csv', type=str, default='my_submission.csv')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='upper epoch limit')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', type=str, default='cuda:1',
                        help='cuda used for training')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for training')
    parser.add_argument('--mode', type=str, default='train',
                        help='choose the mode for runner. mode availiable = train/test')
    parser.add_argument('--pretrain', dest='pretrain', action='store_true',
                        help='use pretrained model')
    parser.add_argument('--alldata', dest='alldata', action='store_true',
                        help='use alldata to train')
    parser.add_argument('--model', type=str, default='resnet50',
                        help='model for the task')
    parser.add_argument('--save_path', type=str, default='./resNet50(from_scratch).pth',
                        help='save the model here')
    parser.add_argument('--model_params_path', type=str, default=None,
                        help='use pretrained model from here')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='term added to the denominator to improve numerical stability')
    parser.add_argument('--log_dir', type=str, default='from_scratch_resnet50',
                        help='save logs')
    return parser.parse_args()
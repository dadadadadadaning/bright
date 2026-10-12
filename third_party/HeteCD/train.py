import os
import time
import random
import numpy as np
import torch.nn as nn
import torch.autograd
from skimage import io
from torch import optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
working_path = os.path.dirname(os.path.abspath(__file__))
from utils.utils_fit import train
from utils.loss import CrossEntropyLoss2d, weighted_BCE_logits, ChangeSimilarity,SCA_Loss,FeatureConsistencyLoss
from utils.utils import accuracy, SCDD_eval_all, AverageMeter, get_confuse_matrix, cm2score
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
#Data and model choose
torch.set_num_threads(4)

###############################################
from datasets import RS_ST as RS
# from models.BiSRNet import BiSRNet as Net
from models.hetecd import hetecd as Net
# 生成一个随机5位整数
import math
nums = math.floor(1e5 * random.random())
seed = 222
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
    #  torch.backends.cudnn.deterministic = True
# # 设置随机数种子
setup_seed(seed)
# from models.SSCDl import SSCDl as Net
NET_NAME = 'xiongan_old'
DATA_NAME = 'xiongan'
EXP_NAME = "EXP"+time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
###############################################    
#Training options
###############################################
args = {
    'net_name': NET_NAME,
    'data_name': DATA_NAME,
    'exp_name':EXP_NAME,
    'train_batch_size': 8,
    'val_batch_size': 8,
    'dice': True,
    'lr': 0.0005,
    'epochs': 200,
    'gpu': True,
    'lr_decay_power': 1.5,
    'weight_decay': 1e-2,
    'momentum': 0.9,
    'print_freq': 5,
    'predict_step': 5,
    'pred_dir': os.path.join(working_path, 'checkpoints',  NET_NAME,DATA_NAME,EXP_NAME,"vis"),
    'chkpt_dir': os.path.join(working_path, 'checkpoints', NET_NAME, DATA_NAME,EXP_NAME),
    'log_dir': os.path.join(working_path, 'checkpoints', NET_NAME,DATA_NAME, EXP_NAME),
    'load_path': os.path.join(working_path, 'checkpoints', DATA_NAME, 'pretrained.pth')
}
###############################################

if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])
# writer = SummaryWriter(args['log_dir'])
#日期时间作为日志文件名
args['log_name']="/log.txt"
def main():        
    net = Net(3).cuda()
    net = nn.DataParallel(net)
    model_dict      = net.state_dict()
    pretrained_dict = torch.load("backbone_weights.pth")
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if "backbone" in k:
            k1 = k.replace("backbone.", "backbone1.")
            k2 = k.replace("backbone.", "backbone2.")
            if k1 in model_dict.keys() and np.shape(model_dict[k1]) == np.shape(v):
                temp_dict[k1] = v
                load_key.append(k1)
            if k2 in model_dict.keys() and np.shape(model_dict[k2]) == np.shape(v):
                temp_dict[k2] = v
                load_key.append(k2)

        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    net.load_state_dict(model_dict)
#------------------------------------------------------#
#   显示没有匹配上的Key
#------------------------------------------------------#
    print("\nSuccessful Load Key:", str(load_key)[:5000], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:5000], "……\nFail To Load Key num:", len(no_load_key))
    model_dict.update(temp_dict)
    net.load_state_dict(model_dict)
        
    train_set_change = RS.Data('train', random_flip=True)
    train_loader_change = DataLoader(train_set_change, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)
    train_set_unchange = RS.Data('train', random_flip=True)
    train_loader_unchange = DataLoader(train_set_unchange, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)
    val_set = RS.Data('val')
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=4, shuffle=False)
    
    criterion = CrossEntropyLoss2d(ignore_index=0).cuda()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'], betas=(0.9, 0.999),weight_decay=0.01, )
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95, last_epoch=-1)

    train(train_loader_change, train_loader_unchange,net, criterion, optimizer, scheduler, val_loader, args)
    print('Training finished.')

        
if __name__ == '__main__':
    main()
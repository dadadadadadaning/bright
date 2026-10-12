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
import copy
from utils.loss import CE_Loss, AlignmentLoss, ChangeSimilarity,Dice_loss, KLDivergenceLoss,FeatureConsistencyLoss
from utils.utils import accuracy, SCDD_eval_all, AverageMeter, get_confuse_matrix, cm2score
import torch.nn.functional as FF
def train(train_loader,train_loader_unchange, net, criterion, optimizer, scheduler, val_loader, args):
    NET_NAME = args['net_name']
    bestiouT=0
    bestiou=0.0
    bestloss=1.0
    bestaccV = 0.0
    begin_time = time.time()
    all_iters = float(len(train_loader)*args['epochs'])
    al_loss = AlignmentLoss()
    # kl_loss = KLDivergenceLoss()
    fc_loss = FeatureConsistencyLoss()
    # criterion_sc = SCA_Loss().cuda()
    curr_epoch=0
    while True:
        net.train()
        #freeze_model(net.FCN)
        start = time.time()
        train_loss = AverageMeter()
        curr_iter = curr_epoch*len(train_loader)
        preds_all = []
        labels_all = []
        names_all = []
        
        # # print(i)
        for i, data in enumerate(train_loader):
            running_iter = curr_iter+i+1
            adjust_lr(optimizer, running_iter, all_iters,args)
            imgs_A, imgs_B, labels, names = data
            # print(imgs_A.shape, imgs_B.shape, labels.shape)
            # print(names)
            # print(imgs_A.shape, imgs_B.shape, labels.shape)
            if args['gpu']:
                imgs_A = imgs_A.cuda().float()
                imgs_B = imgs_B.cuda().float()
                labels = labels.cuda().long()
                # labels = labels>10
            optimizer.zero_grad()
            out_change,features = net(imgs_A, imgs_B)
            out_change = FF.interpolate(out_change, size=(512,512), mode='bilinear', align_corners=True)
            # print(out_change.shape)
            # print(labels.shape)
            cls_weights = torch.tensor([1.0, 1.0]).cuda()
            loss_ce = CE_Loss(out_change, labels, cls_weights=cls_weights)
            align_loss = al_loss(features)
            # sim_loss = fc_loss(features,labels)
            # kl_losss = kl_loss(features)
            if args['dice']:
                # print(out_change.shape, labels.shape)
                loss_dice = Dice_loss(out_change, labels)
                loss =  loss_ce + loss_dice
            else:
                loss =  loss_ce
            loss = loss  + (1/(curr_epoch+1))*2*align_loss
            loss = loss
            loss.backward()
            optimizer.step()
            preds = torch.argmax(out_change, dim=1)
            pred_numpy = preds.cpu().numpy()
            labels_numpy = labels.cpu().numpy()

            preds_all.append(pred_numpy)
            labels_all.append(labels_numpy)
            names_all.extend(names)
            # hist = get_confuse_matrix(2,labels,preds.cpu().numpy())
            # score = cm2score(hist)
            train_loss.update(loss.cpu().detach().numpy())
            curr_time = time.time() - start
            # if (i + 1) % args['print_freq'] == 0:
            #     print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [loss %.4f] [score %s]' % (
            #         curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
            #         train_loss.val, {key: score[key] for key in score})) 
            #     with open(os.path.join(args['log_dir'] + args['log_name']), 'a') as f:
            #         f.write('[epoch %d] [iter %d / %d %.1fs] [lr %f] [bn_loss %.4f] [score %s]\n' % (
            #                         curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
            #                         train_loss.val, {key: score[key] for key in score}))
            
        # if curr_epoch>200 and curr_epoch<260:
        #     for i, data in enumerate(train_loader_unchange):
        #         running_iter = curr_iter+i+1
        #         adjust_lr(optimizer, running_iter, all_iters,args)
        #         imgs_A, imgs_B, labels, names = data
        #         # print(names)
        #         # print(imgs_A.shape, imgs_B.shape, labels.shape)
        #         if args['gpu']:
        #             imgs_A = imgs_A.cuda().float()
        #             imgs_B = imgs_B.cuda().float()
        #             labels = labels.cuda().long()
        #             # labels = labels>10
        #         optimizer.zero_grad()
        #         out_change,features = net(imgs_A, imgs_B)
        #         # out_change_neg,features_neg = net(imgs_A, torch.roll(imgs_B, shifts=-1, dims=0),)
        #         # ones_tensor = torch.ones(out_change.size(0), out_change.size(2), out_change.size(3)).cuda().long()
        #         # zeros_tensor = torch.zeros(out_change.size(0), out_change.size(2), out_change.size(3)).cuda().long()
        #         align_loss = kl_loss(features)
        #         loss_ce = CE_Loss(out_change, labels)
                
        #         if args['dice']:
        #             loss_dice = Dice_loss(out_change, labels)
        #             loss_seg =  loss_ce + loss_dice
        #         else:
        #             loss_seg =  loss_ce
        #         loss = align_loss+loss_seg*0.1
        #         loss.backward()
        #         optimizer.step()
        #         curr_time = time.time() - start
        #         # 打印align_loss
        #         if (i + 1) % args['print_freq'] == 0:
        #             print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [loss %.4f] [align_loss %.4f] [seg_loss %.4f]' % (
        #                 curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
        #                 loss, align_loss, loss_seg))
        #             with open(os.path.join(args['log_dir'] + args['log_name']), 'a') as f:
        #                 f.write('[epoch %d] [iter %d / %d %.1fs] [lr %f] [bn_loss %.4f] [align_loss %.4f] [seg_loss %.4f]\n' % (
        #                                 curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
        #                                 loss, align_loss, loss_seg))
        
        preds_all = np.concatenate(preds_all, axis=0)
        labels_all = np.concatenate(labels_all, axis=0)   
        score_train = cm2score(get_confuse_matrix(2,labels_all, preds_all))     
        with open(os.path.join(args['log_dir'] + args['log_name']), 'a') as f:
            f.write('Epoch: %d  Total time: %.1fs  Train loss %.4f  score %s\n' %(curr_epoch, time.time()-begin_time,train_loss.average(),{key: score_train[key] for key in score_train}))
        print('Epoch: %d  Total time: %.1fs  Train loss %.4f  score %s' %(curr_epoch, time.time()-begin_time, train_loss.average(),{key: score_train[key] for key in score_train}))

        if score_train['iou_1']>bestiouT:
            bestiouT = score_train['iou_1']
            for pred, label, name in zip(preds_all, labels_all, names_all):
                pred = pred.astype(np.uint8)*255
                label = label.astype(np.uint8)*255
                vis_img = np.concatenate([pred, label], axis=1)
                io.imsave(os.path.join(args['pred_dir'], "train_"+name), vis_img)
        score_val,loss_val, val_preds, val_labels, val_names = validate(val_loader, net, criterion, curr_epoch,args)
        # if acc_meter.avg>bestaccT: bestaccT=acc_meter.avg
        if score_val['iou_1']>bestiou:
            bestiou = score_val['iou_1']
            bestloss = loss_val
            torch.save(net.state_dict(), os.path.join(args['chkpt_dir'], NET_NAME+'_%dIoU%.2f'\
                %(curr_epoch, score_val['iou_1']*100)))
            for pred, label, name in zip(val_preds, val_labels, val_names):
                pred = pred.astype(np.uint8)*255
                label = label.astype(np.uint8)*255
                vis_img = np.concatenate([pred, label], axis=1)
                io.imsave(os.path.join(args['pred_dir'], "val_"+name), vis_img)
        with open(os.path.join(args['log_dir'] + args['log_name']), 'a') as f:
            f.write('Epoch: %d  Total time: %.1fs  Val iou %.2f  loss %.4f\n' %(curr_epoch, time.time()-begin_time, bestiou*100, bestloss))
        print('Epoch: %d  Total time: %.1fs  Val iou %.2f  loss %.4f' %(curr_epoch, time.time()-begin_time, bestiou*100, bestloss))
        curr_epoch += 1
        #scheduler.step()
        if curr_epoch >= args['epochs']:
            return

def validate(val_loader, net, criterion, curr_epoch, args):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()


    preds_all = []
    labels_all = []
    names_all = []
    for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, labels, names = data
        
        cls_weights = torch.tensor([0.1, 0.9]).cuda()
        if args['gpu']:
            imgs_A = imgs_A.cuda().float()
            imgs_B = imgs_B.cuda().float()
            labels = labels.cuda().long()
        with torch.no_grad():
            out_change,features = net(imgs_A, imgs_B)
            out_change = FF.interpolate(out_change, size=(512,512), mode='bilinear', align_corners=True)
            loss_bn = CE_Loss(out_change, labels,cls_weights)
            loss =  loss_bn
        val_loss.update(loss.cpu().detach().numpy())
        
        preds = torch.argmax(out_change, dim=1)
        pred_numpy = preds.cpu().numpy()
        labels_numpy = labels.cpu().numpy()
        preds_all.append(pred_numpy)
        labels_all.append(labels_numpy)
        names_all.extend(names)
        # if curr_epoch%args['predict_step']==0:
        #     # print(preds.shape)
        #     for pred, label, name in zip(preds, labels, names):
        #         pred = pred.cpu().numpy().astype(np.uint8)*255
        #         label = label.cpu().numpy().astype(np.uint8)*255
        #         vis_img = np.concatenate([pred, label], axis=1)
        #         io.imsave(os.path.join(args['pred_dir'], "val_"+name), vis_img)
        #         # io.imsave(os.path.join(args['pred_dir'], NET_NAME+'.png'), pred.cpu().numpy()*255)
        #     print('Prediction saved!')
    
    # Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, 2)
    preds_all = np.concatenate(preds_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    hist = get_confuse_matrix(2,labels_all,preds_all)
    score = cm2score(hist)
    curr_time = time.time() - start
    with open(os.path.join(args['log_dir'] + args['log_name']), 'a') as f:
        f.write('Epoch: %d  %.1fs Val loss: %.2f  score: %s\n'\
            %(curr_epoch, curr_time, val_loss.average(),{key: score[key] for key in score}))
    print('Epoch: %d  %.1fs Val loss: %.2f  score: %s'\
    %(curr_epoch, curr_time, val_loss.average(),{key: score[key] for key in score}))

    return score, val_loss.average(), preds_all, labels_all, names_all

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()            

def adjust_lr(optimizer, curr_iter, all_iter, args):
    init_lr=args['lr']
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** args['lr_decay_power'])
    running_lr = init_lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr
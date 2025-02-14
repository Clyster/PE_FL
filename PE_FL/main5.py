import argparse
import os
import copy
import random
import numpy
import math

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import time


from dataloaders.kitti_loader4 import load_calib, input_options, KittiDepth
from metrics import AverageMeter, Result
import criteria
import helper
import vis_utils

from model4 import ENet  # 3 branch
from model2 import Subnet_1, Subnet_2, Subnet_3, Subnet_4
import heapq

parser = argparse.ArgumentParser(description='Sparse-to-Dense')
parser.add_argument('-n',
                    '--network-model',
                    type=str,
                    default="e",
                    choices=["e", "pe"],
                    help='choose a model: enet or penet'
                    )
parser.add_argument('--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',
                    default=100,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--start-epoch-bias',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number bias(useful on restarts)')
parser.add_argument('-c',
                    '--criterion',
                    metavar='LOSS',
                    default='l2',
                    choices=criteria.loss_names,
                    help='loss function: | '.join(criteria.loss_names) +
                    ' (default: l2)')
parser.add_argument('-b',
                    '--batch-size',
                    default=1,
                    type=int,
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=1e-3,
                    type=float,
                    metavar='LR',
                    help='initial learning rate (default 1e-5)')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=1e-6,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 0)')
parser.add_argument('--print-freq',
                    '-p',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
########################
parser.add_argument('--round',
                    type=int,
                    default = 1,
                    help="simulate which client to train(from 1 to 10)"
                    )
#################
parser.add_argument('--data-folder',
                    default='/home/ubuntu-user/Desktop/KITTI/kitti_depth/depth',
                    type=str,
                    metavar='PATH',
                    help='data folder (default: none)')
parser.add_argument('--data-folder-rgb',
                    default='/home/ubuntu-user/Desktop/KITTI/kitti_raw',
                    type=str,
                    metavar='PATH',
                    help='data folder rgb (default: none)')
parser.add_argument('--data-folder-save',
                    default='/home/ubuntu-user/Desktop/KITTI/kitti_depth/submit_test/',
                    type=str,
                    metavar='PATH',
                    help='data folder test results(default: none)')
parser.add_argument('-i',
                    '--input',
                    type=str,
                    default='rgbd',
                    choices=input_options,
                    help='input: | '.join(input_options))
parser.add_argument('--val',
                    type=str,
                    default="full",
                    choices=["select", "full"],
                    help='full or select validation set')
parser.add_argument('--jitter',
                    type=float,
                    default=0.1,
                    help='color jitter for images')
parser.add_argument('--rank-metric',
                    type=str,
                    default='rmse',
                    choices=[m for m in dir(Result()) if not m.startswith('_')],
                    help='metrics for which best result is saved')

parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH')
parser.add_argument('-f', '--freeze-backbone', action="store_true", default=False,
                    help='freeze parameters in backbone')
parser.add_argument('--test', action="store_true", default=False,
                    help='save result kitti test dataset for submission')
parser.add_argument('--cpu', action="store_true", default=False, help='run on cpu')

#random cropping
parser.add_argument('--not-random-crop', action="store_true", default=False,
                    help='prohibit random cropping')
parser.add_argument('-he', '--random-crop-height', default=320, type=int, metavar='N',
                    help='random crop height')
parser.add_argument('-w', '--random-crop-width', default=1216, type=int, metavar='N',
                    help='random crop height')

#geometric encoding
parser.add_argument('-co', '--convolutional-layer-encoding', default="xyz", type=str,
                    choices=["std", "z", "uv", "xyz"],
                    help='information concatenated in encoder convolutional layers')

#dilated rate of DA-CSPN++
parser.add_argument('-d', '--dilation-rate', default="2", type=int,
                    choices=[1, 2, 4],
                    help='CSPN++ dilation rate')

args = parser.parse_args()
args.result = os.path.join('..', 'results')
args.use_rgb = ('rgb' in args.input)
args.use_d = 'd' in args.input
args.use_g = 'g' in args.input
args.val_h = 352
args.val_w = 1216
print(args)

cuda = torch.cuda.is_available() and not args.cpu
if cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("=> using '{}' for computation.".format(device))

# define l
# 
# .oss functions
depth_criterion = criteria.Huber_Combine() 
cl_criterion = criteria.info_nce_loss_depth_maps() 

scaler = torch.cuda.amp.GradScaler()  #

#multi batch
multi_batch_size = 1
def iterate(mode, args, loader, model, optimizer, logger, epoch):
    actual_epoch = epoch - args.start_epoch + args.start_epoch_bias

    block_average_meter = AverageMeter()
    block_average_meter.reset(False)
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]

    # switch to appropriate mode
    assert mode in ["train", "val", "eval", "test_prediction", "test_completion"], \
        "unsupported mode: {}".format(mode)
    if mode == 'train':
        model.train()
        lr = helper.adjust_learning_rate(args.lr, optimizer, actual_epoch, args)
    else:
        model.eval()
        lr = 0

    torch.cuda.empty_cache()
    total_batches = len(loader)
    stop_k = 1
    if epoch in range(0, 2):
        stop_k = 0.6
    elif epoch in range(2, 4):
        stop_k = 0.7
    elif epoch in range(4, 6):
        stop_k = 0.8
    elif epoch in range(6, 8):
        stop_k = 0.9
    else:
        stop_k = 1
    for i, batch_data in enumerate(loader):
        if i >= stop_k * total_batches:
            print("Reach the limit of {} total data, break traing".format(stop_k))
            break
        dstart = time.time()
        batch_data = {
            key: val.to(device)
            for key, val in batch_data.items() if val is not None
        }            

        gt = batch_data[
            'gt'] if mode != 'test_prediction' and mode != 'test_completion' else None
        data_time = time.time() - dstart

        pred = None
        start = None
        gpu_time = 0

        #start = time.time()
        #pred = model(batch_data)
        #gpu_time = time.time() - start

        #'''
        if(args.network_model == 'e'):
            start = time.time()
            with torch.cuda.amp.autocast():
                rgb_pred, depth1_pred, depth2_pred, pred = model(batch_data)
        else:
            start = time.time()
            pred = model(batch_data)

        if(args.evaluate):
            gpu_time = time.time() - start
        #'''

        depth_loss, photometric_loss, smooth_loss, mask = 0, 0, 0, None

        # inter loss_param
        cl_loss, loss = 0, 0

        # round1, round2, round3 = 0, 0, None   # 1, 3, None
        # if(actual_epoch <= round1):
        #     w_st1, w_st2 = 0.2, 0.2
        # elif(actual_epoch <= round2):
        #     w_st1, w_st2 = 0.05, 0.05
        # else:
        #     w_st1, w_st2 = 0, 0

        if mode == 'train':
            # Loss 1: the direct depth supervision from ground truth label
            # mask=1 indicates that a pixel does not ground truth labels
            depth_loss = depth_criterion(pred, gt)
            cl_loss = cl_criterion(depth1_pred, depth2_pred)
            if args.network_model == 'e':
                loss = cl_loss + depth_loss
            else:
                loss = depth_loss

            if i % multi_batch_size == 0:
                optimizer.zero_grad()
            scaler.scale(loss).backward()

            if i % multi_batch_size == (multi_batch_size-1) or i==(len(loader)-1):
                #optimizer.step()
                scaler.step(optimizer)
                scaler.update()
            print("loss:", loss, " epoch:", epoch, " ", i, "/", len(loader))

        if mode == "test_completion":
            str_i = str(i)
            path_i = str_i.zfill(10) + '.png'
            path = os.path.join(args.data_folder_save, path_i)
            vis_utils.save_depth_as_uint16png_upload(pred, path)

        if(not args.evaluate):
            gpu_time = time.time() - start
        # measure accuracy and record loss
        with torch.no_grad():
            mini_batch_size = next(iter(batch_data.values())).size(0)
            result = Result()
            if mode != 'test_prediction' and mode != 'test_completion':
                result.evaluate(pred.data, gt.data, photometric_loss)
                [
                    m.update(result, gpu_time, data_time, mini_batch_size)
                    for m in meters
                ]

                if mode != 'train':
                    logger.conditional_print(mode, i, epoch, lr, len(loader),
                                     block_average_meter, average_meter)
                logger.conditional_save_img_comparison(mode, i, batch_data, pred,
                                                   epoch)
                logger.conditional_save_pred(mode, i, pred, epoch)

    avg = logger.conditional_save_info(mode, average_meter, epoch)
    is_best = logger.rank_conditional_save_best(mode, avg, epoch)
    if is_best and not (mode == "train"):
        logger.save_img_comparison_as_best(mode, epoch)
    logger.conditional_summarize(mode, avg, is_best)

    return avg, is_best

def average_weight(w):
    length=[]
    for i in w:
        tmp_len = len(i)
        length.append(tmp_len)
    max_value = max(length)  # find the longest len of all file
    max_index = length.index(max_value) # locate the longest file
    w_avg = copy.deepcopy(w[max_index])   # choose the longest weight file as base
    for key in w_avg.keys():
        count =1
        for i in range(0, len(w)):
            if i==max_index:
                continue
            if key in w[i]:
                w_avg[key] += w[i][key]
                count = count + 1
        w_avg[key] = torch.div(w_avg[key],count)
    return w_avg

def calculate_k(idx, all):
    selected_values = [all[index-1] for index in idx]
    exp_values = [math.exp(value) for value in selected_values]
    total = sum(exp_values)
    s1 = [total / exp_value for exp_value in exp_values]
    total2 = sum(s1)
    u_want  = [s / total2 for s in s1]
    return u_want
    
def Weighted_weight(w, users, userState):
    length=[]
    k = calculate_k(users, userState)
    for i in w:
        tmp_len = len(i)
        length.append(tmp_len)
    max_value = max(length)  # find the longest len of all file
    max_index = length.index(max_value) # locate the longest file
    w_Weighted = copy.deepcopy(w[max_index])   # choose the longest weight file as base
    for key in w_Weighted.keys():
        w_Weighted[key] = 0
        for i in range(0, len(w)):
            if key in w[i]:
                if i==max_index:
                    # w_Weighted[key] -= (float(w[i][key]) - (float(w[i][key]) * float(k[i])))
                    w_Weighted[key] += (w[i][key] * float(k[i]))
                # else:
                #     w_Weighted[key] += (w[i][key] * float(k[i]))
        w_Weighted[key] = torch.div(w_Weighted[key],len(w))
    return w_Weighted

def random_lost():
    a = False
    b = False
    lst = [True, False]
    arr = numpy.array(lst)
    a, b = numpy.random.choice(arr,2,replace=False)
    return a, b

def calcualte_diff(w1, w2, limit):
    p1 = torch.cat([v.view(-1) for v in w1.values()])
    p2 = torch.cat([v.view(-1) for v in w2.values()])
    num_params = p1.size(0)
    lm = min(limit, num_params)
    total_diff = (int(torch.sum(torch.abs(p1[:lm]-p2[:lm])).item())/400)
    return total_diff

def getUser(userState, userNum):
    selected_indices = []
    while len(selected_indices) < userNum:
        chosen_idx = random.choices(range(len(userState)), weights=userState, k=1)[0] + 1
        if (chosen_idx not in selected_indices) and (chosen_idx > 10):    #####  
            selected_indices.append(chosen_idx)
    return selected_indices

        

# simulate modal missing
args.d_lost = False
args.rgb_lost = False  

def main():
    global args
    # global d_lost
    # global rgb_lost
    
    checkpoint = None
    is_eval = False
    parts_idx = range(1,51)
    global_weights =[]
    user_state = [499] * 50  # to save every user's abs weight diff to global weights

    if args.evaluate:
        args_new = args
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}' ... ".format(args.evaluate),
                end='')
            checkpoint = torch.load(args.evaluate, map_location=device)
            # args = checkpoint['args']
            # args.start_epoch = checkpoint['epoch'] + 1
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            is_eval = True

            print("Completed.")
        else:
            is_eval = True
            print("No model found at '{}'".format(args.evaluate))
            #return

    if args.resume:  # optionally resume from a checkpoint
            args_new = args
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}' ... ".format(args.resume),
                    end='')
                checkpoint = torch.load(args.resume, map_location=device)
                global_weights = checkpoint['model']
                
                args.start_epoch = checkpoint['epoch'] + 1
                args.data_folder = args_new.data_folder
                args.val = args_new.val
                
                print("Completed. Resuming from epoch {}.".format(
                    checkpoint['epoch']))
            else:
                print("No checkpoint found at '{}'".format(args.resume))
                return

    logger = helper.logger(args)
    print("=> logger created.")
    
    val_dataset = KittiDepth('val', args)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True)  # set batch size to be 1 for validation
    print("\t==> val_loader size:{}".format(len(val_loader)))

    if is_eval == True:
        for p in model.parameters():
            p.requires_grad = False

        result, is_best = iterate("val", args, val_loader, model, None, logger,
                            args.start_epoch - 1)
        return

    
       
    
    for global_epoch in range(args.start_epoch,(args.epochs)+1):
        print("Global epoch:",global_epoch)
        users = random.sample(parts_idx, 10)  # debug 1
        # users = getUser(user_state, 10)
        print("this round choose user:", users)
        local_weights =[]
        start_time = time.time()
        user_state_list = []    # to save every user's abs weight diff to global weights
        for user in users:
            
            # modal lost controller
            modal_lost = False
            args.d_lost = False
            args.rgb_lost = False
            # user = 2  # debug only!
            args.round = user
            print("This is {} user-------------".format(args.round))
            # if user in range(1,11):  # 20 only lost depth
            #     print("WARNING! Start simulate modal missing!")
            #     modal_lost = True 
            #     args.d_lost, args.rgb_lost = random_lost()
            #     # args.d_lost = True # debug only!
            #     print("depth lost:{} rgb lost:{}".format(args.d_lost, args.rgb_lost))
            # else:
            #     args.d_lost = False
            #     args.rgb_lost = False
            
            # rebuild model every user
            print("=> creating model and optimizer ... ", end='')
            # model
            # model = ENet(args)
            model = None          
            penet_accelerated = False
            torch.cuda.empty_cache() 
            model = ENet(args).to(device)    # DEBUG
            # if global_epoch in range(0,2): # 0,1
            #     print("subnetwork_1")
            #     model = Subnet_1(args).to(device)
                
            # elif global_epoch in range(2,4): # 2, 3
            #     print("subnetwork_2")
            #     model = Subnet_2(args).to(device)
            
            # elif global_epoch in range(4,6): # 4,5
            #     print("subnetwork_3")
            #     model = Subnet_3(args).to(device)
            
            # elif global_epoch in range(6,8): # 6,7
            #     print("subnetwork_4")
            #     model = Subnet_4(args).to(device)
            # else:
            #     print("Start Enet now!")
            #     model = ENet(args).to(device)
            
            

            
            model_named_params = None
            model_bone_params = None
            model_new_params = None
            # optimizer
            optimizer = None
            if (args.freeze_backbone == True):
                for p in model.backbone.parameters():
                    p.requires_grad = False
                model_named_params = [
                    p for _, p in model.named_parameters() if p.requires_grad
                ]
                optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
            
            else:
                model_named_params = [
                    p for _, p in model.named_parameters() if p.requires_grad
                ]
                optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
            print("completed.")
            
            # Loading global weights before data parallel
            if (len(global_weights) == 0):
                print("global_weights is empty!")
            else:
                try:
                    model.load_state_dict(global_weights)
                    print("Global weights loaded successfully!")
                except RuntimeError as e:
                    print(f"Model Fail to Load: {e}")
                
            # data parallel
            model = torch.nn.DataParallel(model) 
            # Data loading code
            print("=> creating data loaders ... ")
            if not is_eval:
                if modal_lost == True:
                    print("Modal Missing!Dataset will be replaced by zero matrix in iterate()!!!!!!!!!!!!")
                train_dataset = KittiDepth('train', args, epoch=global_epoch)
                train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        num_workers=args.workers,
                                                        pin_memory=True,
                                                        sampler=None)
                            
                print("\t==> train_loader size:{}".format(len(train_loader)))
                

            print("=> starting main loop ...")
            for epoch in range(1, 6):
                print("=> starting training user {} epoch {} ..".format(user,epoch))
                iterate("train", args, train_loader, model, optimizer, logger, global_epoch)  # train for one epoch
                
                for p in model.parameters():
                    p.requires_grad = True
            lw = copy.deepcopy(model.module.state_dict())
            local_weights.append(lw)
            assert lw, "local_weights MISSING!!! Check CODE!!!!!"
            print("local_weights saved! :D)")
            
            if(len(global_weights)!=0):
                stateU = calcualte_diff(lw, global_weights, 10000)
                if stateU < 500:
                    user_state[user-1] = float(stateU)
                else:
                    user_state[user-1] = 500.0
                    
                # user_state[user] = calcualte_diff(lw, global_weights, 10000)
                print("user_state:", user_state[user-1])
                user_state_list.append(user_state[user-1])

        if global_epoch in range(2,10):    # acclearting the convergence
            max_error = max(user_state_list)
            max_index = user_state_list.index(max_error)
            del local_weights[max_index]
        global_weights =average_weight(local_weights)
        # if len(global_weights) !=0:
        #     global_weights = Weighted_weight(local_weights, users= users, userState= user_state)
        #     print("Save Weighted weight")
        # else:
        #     global_weights =average_weight(local_weights)
        #     print("Save average Weight")

        end_time = time.time()
        cost_time = end_time - start_time
        # model.load_state_dict(global_weights)  ###
        # validation memory reset
        for p in model.parameters():
            p.requires_grad = False
        result, is_best = iterate("val", args, val_loader, model, None, logger, global_epoch)  # evaluate on validation set
        helper.save_checkpoint({ # save checkpoint
            'epoch': global_epoch,
            'model': model.module.state_dict(),
            'best_result': logger.best_result,
            #'optimizer' : optimizer.state_dict(),
            'args' : args,
        }, is_best, global_epoch, logger.output_directory)
        print("Time consumption:{}".format(cost_time))
        print("Global weights: global epoch {} Saved!".format(global_epoch))


if __name__ == '__main__':
    main()
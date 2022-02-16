# import statements
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import os
import time
import datetime
import logging
import tqdm
import src.model as model
from src.dataset import Graph
from torch.utils.tensorboard import SummaryWriter
from src.losses import *
import argparse
import src.utils as utils
from src.dataset.pose_dataset import PoseDataset
from src.optimizer import Optimizer

# set flags / seeds
torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Start with main code
if __name__ == '__main__':
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(description="Train a network for ")
    
    parser.add_argument('--config', '-c', type=str, default='', help='ID of the using config', required=True)
    parser.add_argument('--gpus', '-g', type=str, nargs='+', default='1', help='Using GPUs')
    opt = parser.parse_args() 


    config = utils.get_config(opt.config)
    print("==== Config: ", config, sep='\n')

    # set flags / seeds
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

    torch.backends.cudnn.benchmark = True
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    # add code for datasets (we always use train and validation/ test set)
    graph = Graph(config.dataset)
    train_dataset = PoseDataset(
        root=config.dataset_args[config.dataset]["train_data"],
        inputs=config.dataset_args[config.dataset]["inputs"],
        num_frame=config.dataset_args[config.dataset]["num_frame"],
        connect_joint=graph.connect_joint,
        is_train=True)

    print("===== Train dataset:", len(train_dataset))
    train_data_loader = data.DataLoader(train_dataset,
        batch_size=config.dataset_args[config.dataset]["train_batch_size"],
        num_workers=4*len(opt.gpus),
        pin_memory=True, shuffle=True, drop_last=True)
    
    val_dataset = PoseDataset(
        root=config.dataset_args[config.dataset]["val_data"],
        inputs=config.dataset_args[config.dataset]["inputs"],
        num_frame=config.dataset_args[config.dataset]["num_frame"],
        connect_joint=graph.connect_joint,
        is_train=True)

    print("===== Val dataset:", len(val_dataset))

    val_dataset = data.DataLoader(val_dataset,
        batch_size=config.dataset_args[config.dataset]["eval_batch_size"], 
        num_workers=4*len(opt.gpus),
        pin_memory=True, shuffle=True, drop_last=True)
    
    # instantiate network (which has been imported from *networks.py*)
    net = model.create(config, graph)
    
    # create losses (criterion in pytorch)
    criterion = TripletLoss()
    
    # if running on GPU and we want to use cuda move model there
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
    
    # create optimizers
    optimizer = Optimizer(config.optimizer_args[config.optimizer])
    optim = optimizer.get_optimizer(net, config.optimizer)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, 
        step_size=config.scheduler_args["step"]["step_size"],
        gamma=config.scheduler_args["step"]["gamma"])
    
    # load checkpoint if needed/ wanted
    start_epoch = 0
    net = torch.nn.DataParallel(net)    
    if config.pretrained is not None:
        ckpt = torch.load(config.pretrained, map_location=torch.device('cuda:0'))
        # print(ckpt["epoch"], ckpt["loss"])
        net.load_state_dict(ckpt['model'])
        print("last checkpoint restored")
        
        
    # typically we use tensorboardX to keep track of experiments
    writer = SummaryWriter()
    
    # now we start the main loop
    max_epoch = config.epochs
    prev_loss = 1000
    for epoch in range(start_epoch, max_epoch):
        # set models to train mode
        net.train()        
        # use prefetch_generator and tqdm for iterating through data
        pbar = enumerate(train_data_loader)
        start_time = time.time()
        
        running_loss = 0.0
        # for loop going through dataset
        for i, data in tqdm.tqdm(pbar):
            # data preparation
            anchor, positive, negative = data
            prepare_time = start_time - time.time()
            if use_cuda:
                anchor = anchor.cuda().float()
                positive = positive.cuda().float()
                negative = negative.cuda().float()
            
            # It's very good practice to keep track of preparation time and computation time using tqdm to find any issues in your dataloader
            # forward and backward pass
            optim.zero_grad()
            anchor_embedded = net(anchor)
            positive_embbeded = net(positive)
            negative_embbedded = net(negative)

            loss = criterion(anchor_embedded, positive_embbeded, negative_embbedded)
            loss.backward()
            optim.step()
            scheduler.step()
            running_loss += loss.item()
            process_time = start_time - time.time() - prepare_time

            # print("Compute efficiency: {:.2f}, epoch: {}/{}:".format(
            #     process_time/(process_time+prepare_time), epoch, max_epoch))
        running_loss = (running_loss / (i + 1))
        print("Epoch: {}, loss: {}".format(epoch, running_loss))

        # udpate tensorboardX
        writer.add_scalar('Loss/train', running_loss, epoch)
        writer.add_scalar('Learning rate', optim.param_groups[0]['lr'], epoch)
            
        # do a test pass every x epochs
        if epoch % config.val_scheduler == config.val_scheduler - 1:
            val_loss = 0
            net.eval()
            pbar = enumerate(val_dataset)
            for i, data in pbar:
            # data preparation
                anchor, positive, negative = data
                if use_cuda:
                    anchor = anchor.cuda().float()
                    positive = positive.cuda().float()
                    negative = negative.cuda().float()
                
                anchor_embedded = net(anchor)
                positive_embbeded = net(positive)
                negative_embbedded = net(negative)

                loss = criterion(anchor_embedded, positive_embbeded, negative_embbedded)
                val_loss += loss.item()
                process_time = start_time - time.time() - prepare_time
            val_loss = val_loss / (i + 1)
            print("Epoch: {}, val loss: {}".format(epoch, val_loss))
            writer.add_scalar('Loss/val', val_loss, epoch)

        if val_loss <= prev_loss:
            # save checkpoint if needed
            print("Val Loss decreased from {} to {}".format(prev_loss, val_loss))
            utils.save_checkpoint(net, optim, running_loss, epoch, config.save_path, opt.config)
            prev_loss = val_loss
        print("=========================================")
            
            
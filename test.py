# import statements
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import os
import time
import src.model as model
from src.dataset import Graph
from torch.utils.tensorboard import SummaryWriter
from src.losses import *
import argparse
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import src.utils as utils
from src.dataset.pose_dataset import PoseDataset
from src.optimizer import Optimizer

# set flags / seeds
torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

def knn(query, gallery, k=5):
    """
    Args:
        - gallery: (N, 128)
        - query: (1, 128)
        - k: number of nearest neighbors

    Return:
        - sorted_dist: top k minimum distance
        - indices: top k indices 
    """
    query_c = torch.repeat_interleave(query, repeats=gallery.shape[0], dim=0)
    distances = criterion.distance(query_c, gallery)
    sorted_dist, indices = torch.sort(distances, dim=-1)
    return sorted_dist[:k], indices[:k]

# Start with main code
if __name__ == '__main__':
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(description="Train a network for ")
    
    parser.add_argument('--config', '-c', type=str, default='', help='ID of the using config', required=True)
    parser.add_argument('--gpus', '-g', type=str, nargs='+', default='1', help='Using GPUs')
    opt = parser.parse_args() 

    config = utils.get_config(opt.config)

    # set flags / seeds
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

    torch.backends.cudnn.benchmark = True
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    # add code for datasets (we always use train and validation/ test set)
    graph = Graph(config.dataset)
    gallery_dataset = PoseDataset(
        root="data/support_set",
        inputs=config.dataset_args[config.dataset]["inputs"],
        num_frame=config.dataset_args[config.dataset]["num_frame"],
        connect_joint=graph.connect_joint,
        is_train=False)

    gallery_data_loader = data.DataLoader(gallery_dataset,
        batch_size=config.dataset_args[config.dataset]["train_batch_size"],
        num_workers=4*len(opt.gpus),
        pin_memory=True, shuffle=True, drop_last=True)
    
    val_dataset = PoseDataset(
        root=config.dataset_args[config.dataset]["val_data"],
        inputs=config.dataset_args[config.dataset]["inputs"],
        num_frame=config.dataset_args[config.dataset]["num_frame"],
        connect_joint=graph.connect_joint,
        is_train=False)

    val_dataset = data.DataLoader(gallery_dataset,
        batch_size=1, 
        num_workers=4*len(opt.gpus),
        pin_memory=True, shuffle=False, drop_last=True)
    
    net = model.create(config, graph)
    
    criterion = TripletLoss()

    if config.pretrained is not None:
        ckpt = torch.load(config.pretrained, map_location=torch.device('cuda:0'))
        # print(ckpt["epoch"], ckpt["loss"])
        net.load_state_dict(ckpt['model'])
        print("last checkpoint restored")
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()

    net = torch.nn.DataParallel(net)    
    net.eval()
    pbar = enumerate(gallery_data_loader)

    # create gallery
    gallery_data = []
    gallery_label = []

    y_pred = []
    y_true = []

    for i, data in pbar:        
        data, label = data
        if use_cuda:
            data = data.cuda().float()    
        data_embedded = net(data)
        gallery_data.append(data_embedded)
        gallery_label.append(label)

    gallery_label = torch.cat(gallery_label).tolist()
    gallery_data = torch.cat(gallery_data)

    # # calculate accuracy
    pbar = enumerate(val_dataset)
    for i, query in pbar:
        data, label = query
        label = label.item()
        if use_cuda:
            data = data.cuda().float()
        
        data_embedded = net(data)
        sorted_dist, indices = knn(data_embedded, gallery_data)
        sorted_dist, indices = sorted_dist.tolist(), indices.tolist()
        pred_k = list(map(lambda x: gallery_label[x], indices))
        y_pred.append(max(pred_k, key=pred_k.count))
        y_true.append(label)

    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
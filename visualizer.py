import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
from torch.utils import data
import src.utils as utils
import argparse
import os
import numpy as np
import pandas as pd
import src.model as model
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import src.dataset
from src.dataset import Graph
from src.dataset.pose_dataset import PoseDataset
# from sklearn.decomposition import PCA

# class Visualizer:
#     #TODO: t-NSE
#     def __init__(self):
#         pass

#     def create_embedding(self):
#         pass

#     def create_label(self):
#         pass

#     def visualize(self):
#         pass

parser = argparse.ArgumentParser(description='PyTorch t-SNE for STL10')
parser.add_argument('--save-dir', type=str, default='./results', help='path to save the t-sne image')
parser.add_argument('--batch-size', type=int, default=128, help='batch size (default: 128)')
parser.add_argument('--seed', type=int, default=1, help='random seed value (default: 1)')
parser.add_argument('--config', '-c', type=str, default='', help='ID of the using config', required=True)
parser.add_argument('--gpus', '-g', type=str, nargs='+', default='1', help='Using GPUs')
args = parser.parse_args() 
config = utils.get_config(args.config)
print("==== Config: ", config, sep='\n')

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set seed
torch.manual_seed(args.seed)
if device == 'cuda':
    torch.cuda.manual_seed(args.seed)

# set model
graph = Graph(config.dataset)
net = model.create(config, graph)

if config.pretrained is not None:
    ckpt = torch.load(config.pretrained, map_location=torch.device("cuda:0"))
    net.load_state_dict(ckpt["model"])
    print("=================last checkpoint restored")

if device == 'cuda':
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

dataset = PoseDataset(
    root=config.dataset_args[config.dataset]["val_data"],
    inputs=config.dataset_args[config.dataset]["inputs"],
    num_frame=config.dataset_args[config.dataset]["num_frame"],
    connect_joint=graph.connect_joint,
    is_train=False)

print("===== Dataset:", len(dataset))
dataloader = data.DataLoader(dataset,
    batch_size=config.dataset_args[config.dataset]["train_batch_size"],
    num_workers=4*len(args.gpus),
    pin_memory=True, shuffle=True, drop_last=True)
    

def gen_features():
    net.eval()
    targets_list = []
    outputs_list = []

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            targets_np = targets.data.cpu().numpy()

            outputs = net(inputs)
            outputs_np = outputs.data.cpu().numpy()
            
            targets_list.append(targets_np[:, np.newaxis])
            outputs_list.append(outputs_np)
            
            if ((idx+1) % 10 == 0) or (idx+1 == len(dataloader)):
                print(idx+1, '/', len(dataloader))

    targets = np.concatenate(targets_list, axis=0)
    outputs = np.concatenate(outputs_list, axis=0).astype(np.float64)

    return targets, outputs

def tsne_plot(save_dir, targets, outputs):
    print('generating t-SNE plot...')
    # tsne_output = bh_sne(outputs)
    tsne = TSNE(perplexity=40, n_iter=2000)
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['targets'] = targets

    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='targets',
        palette=sns.color_palette("hls", len(np.unique(targets))),
        data=df,
        marker='o',
        legend="full",
        alpha=1
    )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig(os.path.join(save_dir,'tsne.png'), bbox_inches='tight')
    print('done!')

targets, outputs = gen_features()
tsne_plot(args.save_dir, targets, outputs)
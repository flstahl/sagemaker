import argparse
import ast
import os
import time

# oss
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.distributed as dist
import torch

from utils_waymo import KITTI_ds

from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops.misc import Permute
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import zipfile
import os
import torchvision
import time
from engine import train_one_epoch, evaluate
import utils
import transforms as T
from torch.optim.lr_scheduler import StepLR


from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor

    # NOTE: since the library transforms was changed, we now have to use PILToTensor + ConvertImageDtype instead of ToTensor (as used in Tutorial)

    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def init_distributed():

    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()

    # model & training parameters
    parser.add_argument("--epochs", type=int, default=1)
    # if used, iterations must be smaller than expected iterations in one epoch
    #parser.add_argument("--iterations", type=int, default=10e5)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    #parser.add_argument("--lr_warmup_ratio", type=float, default=1)
    #parser.add_argument("--epoch_peak", type=int, default=2)
    #parser.add_argument("--lr_decay_per_epoch", type=float, default=1)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--log-freq", type=int, default=1)
    parser.add_argument("--eval-size", type=int, default=30)
    #parser.add_argument("--height", type=int, default=1208)
    #parser.add_argument("--width", type=int, default=1920)

    # infra configuration
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--amp", type=str, default="True")

    # Data, model, and output directories
    parser.add_argument("--cache", type=str)
    parser.add_argument("--network", type=str, default="swin_s")
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--checkpoint-dir", type=str, default="/opt/ml/checkpoints")
    parser.add_argument("--dataset", type=str, default=os.environ.get("SM_CHANNEL_DATASET"))
    parser.add_argument("--manifest", type=str, default="manifest.json")
    #parser.add_argument("--val-manifest", type=str, default="val_manifest.json")
    #parser.add_argument("--class-list", type=str, default="class_list.json")
    parser.add_argument("--bucket", type=str, default = "waymo-open-dataset1")

    args, _ = parser.parse_known_args()
    

    print('starting training')

    #root = 'waymo'
    start = time.time()
    
    
    ########################################################
    #add more model backbone options here
    ###########################################################
    
    backbone = torchvision.models.swin_s(weights='IMAGENET1K_V1').features
    backbone = nn.Sequential(backbone, Permute([0,3,1,2]))

    backbone.out_channels = 768
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))


    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    #init_distributed()
    model = FasterRCNN(backbone,
                       num_classes=args.classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    #model = model.cuda()
    #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #local_rank = int(os.environ['LOCAL_RANK'])

    #model =torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    dataset = KITTI_ds(transforms = get_transform(train=True), 
                       s3_bucket = args.bucket, 
                       cache = args.cache, 
                       manifest = os.path.join(args.dataset, args.manifest))
    
    
    dataset_test = KITTI_ds(transforms = get_transform(train=False), 
                       s3_bucket = args.bucket, 
                       cache = args.cache, 
                       manifest = os.path.join(args.dataset, args.manifest))

    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()

    dataset = torch.utils.data.Subset(dataset, indices[:-500])
    
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-500:])

    #train_sampler = DistributedSampler(dataset=dataset,shuffle=True)
    #test_sampler =DistributedSampler(dataset=dataset_test, shuffle=True)
    
    batch_size = args.batch
    
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle = True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    num_classes = args.classes

    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(lr = 0.0001, weight_decay=0.05, params=params)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = args.epochs
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    end = time.time()


    torch.save(model, os.path.join(args.checkpoint_dir, 'swin_s_waymo_dist.pth'))
    print('Training done, took '+  str(end-start) + ' seconds')





    
   
    

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.distributed as dist
import torch

import json
import os

import boto3


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



class KITTI_ds(torch.utils.data.Dataset):

    #CLASSES = ['Car', 'Pedestrian', 'Cyclist', 'DontCare', 'Misc', 'Person_sitting', 'Tram', 'Truck', 'Van']
    #CLASSES = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'DontCare': 4, 'Misc': 5, 'Person_sitting': 6, 'Tram':7, 'Truck': 8, 'Van': 9}

    def __init__(self, s3_bucket, cache, manifest, transforms = None):
        
        with open(os.path.join(manifest)) as file:
            self.manifest = json.load(file)
        
        
        self.CLASSES = self.manifest.pop('classes')
        self.cat2label = {k: i+1 for i, k in enumerate(self.CLASSES)}
        self.record_list = list(self.manifest.items())
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        #self.imgs = list(sorted(os.listdir(os.path.join(root, "image_0"))))
        #self.labels = list(sorted(os.listdir(os.path.join(root, "label_0"))))
        self.bucket = s3_bucket
        self.cache = cache
        self.s3 = boto3.resource("s3")


    def __getitem__(self, idx):

        s3 = self.s3
        
        img_name = self.record_list[idx][1]["image_name"]
        img_key = self.record_list[idx][1]["img_path"]
        
        label_name = self.record_list[idx][1]["label_name"]
        label_key = self.record_list[idx][1]["label_path"]
        
        #img_path = os.path.join("image_0", self.imgs[idx])
        #label_path = os.path.join("label_0", self.labels[idx])
        
        # read file from S3 or locally, if already downloaded
        
        image_local_path = os.path.join(self.cache, img_name)
        if os.path.exists(image_local_path):
            img = Image.open(image_local_path).convert("RGB")
            
        else:
            s3.meta.client.download_file(
                Key=img_key, 
                Filename=image_local_path,
                Bucket=self.bucket)
            img = Image.open(image_local_path).convert("RGB")
            
            
        label_local_path = os.path.join(self.cache, label_name)
        if os.path.exists(label_local_path):
            with open(label_local_path) as file:
                lines = [line.rstrip() for line in file]
            
        else:
            s3.meta.client.download_file(
                Key=label_key, 
                Filename=label_local_path,
                Bucket=self.bucket)
            with open(label_local_path) as file:
                lines = [line.rstrip() for line in file]
        
        
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        #mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        #mask = np.array(mask)
        # instances are encoded as different colors
        #obj_ids = np.unique(mask)
        # first id is the background, so remove it
        #obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        #masks = mask == obj_ids[:, None, None]


        # load annotations

        content = [line.strip().split(' ') for line in lines]

        bbox_names = [x[0] for x in content]

        #bboxes = [[float(info) for info in x[4:8]] for x in content]

        # get bounding box coordinates for each object
        num_objs = len(content)
        bboxes = []

        for x in content:
            xmin = float(x[4])
            xmax = float(x[6])
            ymin = float(x[5])
            ymax = float(x[7])
            bboxes.append([xmin, ymin, xmax, ymax])

            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []

            # filter 'DontCare'
            for bbox_name, bbox in zip(bbox_names, bboxes):
                if bbox_name in self.cat2label:
                    gt_labels.append(self.cat2label[bbox_name])
                    gt_bboxes.append(bbox)
                else:
                    gt_labels_ignore.append(-1)
                    gt_bboxes_ignore.append(bbox)

        # convert everything into a torch.Tensor
        gt_bboxes = torch.as_tensor(gt_bboxes, dtype=torch.float32)
        gt_labels = torch.as_tensor(gt_labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * (gt_bboxes[:, 2] - gt_bboxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(gt_labels),), dtype=torch.int64)

        target = {}
        target["boxes"] = gt_bboxes
        target["labels"] = gt_labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        #target["name"] = str(img_path)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.record_list)



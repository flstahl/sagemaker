import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.distributed as dist
import torch

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


class KITTI_ds(torch.utils.data.Dataset):

    #CLASSES = ['Car', 'Pedestrian', 'Cyclist', 'DontCare', 'Misc', 'Person_sitting', 'Tram', 'Truck', 'Van']
    #CLASSES = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'DontCare': 4, 'Misc': 5, 'Person_sitting': 6, 'Tram':7, 'Truck': 8, 'Van': 9}
    CLASSES = ('VEHICLE', 'PEDESTRIAN', 'CYCLIST', 'SIGN')


    def __init__(self, root, transforms = None):
        self.cat2label = {k: i+1 for i, k in enumerate(self.CLASSES)}
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "image_0"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "label_0"))))

        #ADDING SAFETY MECHANISM TO ENSURE THAT ONLY IDS THAT ARE PRESENT IN BOTH IMAGES AND LABELS ARE USED
        #--> OTHERWISE IINDEX OUT OF BOUNDS ERROR

        imgs_split = [idx.split('.')[0] for idx in self.imgs]
        labels_split = [idx.split('.')[0] for idx in self.labels]
        usable_ids = list(set(imgs_split).intersection(set(labels_split)))

        self.imgs = [idx for idx in self.imgs if idx.split('.')[0] in usable_ids]
        self.labels = [idx for idx in self.labels if idx.split('.')[0] in usable_ids]


    def __getitem__(self, idx):

        # load images
        img_path = os.path.join(self.root, "image_0", self.imgs[idx])
        label_path = os.path.join(self.root, "label_0", self.labels[idx])
        img = Image.open(img_path).convert("RGB")

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

        with open(label_path) as file:
            lines = [line.rstrip() for line in file]
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
        return len(self.imgs)


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

    print('starting training')

    root = 'waymo'
    start = time.time()
    backbone = torchvision.models.swin_v2_s(weights='IMAGENET1K_V1').features
    backbone = nn.Sequential(backbone, Permute([0,3,1,2]))

    backbone.out_channels = 768
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))


    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    #init_distributed()
    model = FasterRCNN(backbone,
                       num_classes=5,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    #model = model.cuda()
    #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #local_rank = int(os.environ['LOCAL_RANK'])

    #model =torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    dataset = KITTI_ds(root, get_transform(train=True))
    dataset_test = KITTI_ds(root, get_transform(train=False))

    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()

    dataset = torch.utils.data.Subset(dataset, indices[:-500])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-500:])

    #train_sampler = DistributedSampler(dataset=dataset,shuffle=True)
    #test_sampler =DistributedSampler(dataset=dataset_test, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=3, num_workers=4,pin_memory=True,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False,
        num_workers=4,collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 5

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lr = 0.0001, weight_decay=0.05, params=params)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = 5
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    end = time.time()


    torch.save(model, 'swin_s_waymo_dist.pth')
    print('Training done, took '+  str(end-start) + ' seconds')
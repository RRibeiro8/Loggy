import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.data as data
from torch.autograd import Variable

from PIL import Image
import os.path
import json

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

#model architecture: alexnet | densenet121 | densenet161 | densenet169 | densenet201 | googlenet | 
#inception_v3 | mnasnet0_5 | mnasnet0_75 | mnasnet1_0 | mnasnet1_3 | mobilenet_v2 | resnet101 | 
#resnet152 | resnet18 | resnet34 | resnet50 | resnext101_32x8d | resnext50_32x4d | shufflenet_v2_x0_5 | 
#shufflenet_v2_x1_0 | shufflenet_v2_x1_5 | shufflenet_v2_x2_0 | squeezenet1_0 | squeezenet1_1 | 
#vgg11 | vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19 | vgg19_bn | wide_resnet101_2 | wide_resnet50_2


best_acc1 = 0

gpu = 0
pretrained = True
net_exists = True
arch = "resnet50"
#print('model architecture: ' + ' | '.join(model_names))
num_classes = 365
lr = 0.01 #learning rate
momentum = 0.9 #momentum
weight_decay = 1e-4 #weight decay (default: 1e-4)
epochs = 30 #number of total epochs to run
start_epoch = 0 #manual epoch number (useful on restarts)
evaluate = False # evaluate model on validation set
resume = "checkpoint.pth.tar"#None #path to latest checkpoint (default: none)
print_freq = 10 #print frequency (default: 10)

train_results = {}
val_results = {}

def main():

    ngpus_per_node = torch.cuda.device_count()

    # Simply call main_worker function
    main_worker(gpu, ngpus_per_node)

def default_loader(path):
    return Image.open(path).convert('RGB')

def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append( (impath, int(imlabel)) )
                    
    return imlist

class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None,
            flist_reader=default_flist_reader, loader=default_loader):
        self.root   = root
        self.imlist = flist_reader(flist)       
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader((self.root + impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self):
        return len(self.imlist)


def main_worker(gpu, ngpus_per_node):
    global best_acc1, start_epoch

    #if gpu is not None:
    print("Use GPU: {} for training".format(torch.cuda.current_device()))

    # create model
    if pretrained:
        print("=> using pre-trained model '{}'".format(arch))
        model = models.__dict__[arch](pretrained=True)#, num_classes=365)
    else:
        if net_exists:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch](num_classes=365)
        else:
            model = Net()

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    #model.classifier._modules['6'] = nn.Linear(4096, num_classes)

    #if gpu is not None:
    torch.cuda.set_device(gpu)
    model = model.cuda()#gpu)
    #else:

    print(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()#gpu)

    optimizer = torch.optim.SGD(model.fc.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    # DataParallel will divide and allocate batch_size to all available GPUs
    if arch.startswith('alexnet') or arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()


    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            if gpu is None:
                checkpoint = torch.load(resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(gpu)
                checkpoint = torch.load(resume, map_location=loc)
            start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    cudnn.benchmark = True

    data = 'data/places365_Standard/'
    # Data loading code
    traindir = os.path.join(data, 'train')
    valdir = os.path.join(data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        ImageFilelist(root="data/places365_Standard/train/", flist="data/places365_Standard/train.txt",
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), normalize,
                ])),
        batch_size=32, shuffle=True,
        num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        ImageFilelist(root="data/places365_Standard/val/", flist="data/places365_Standard/val.txt",
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(), normalize,
                ])),
        batch_size=32, shuffle=False,
        num_workers=4, pin_memory=True)


    if evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(start_epoch, epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

    with open('train.json', 'w') as jsonfile:
        json.dump(train_results, jsonfile, indent=4)

    with open('val.json', 'w') as jsonfile:
        json.dump(val_results, jsonfile, indent=4)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    start_time = time.time()
    
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #if gpu is not None:
        images = Variable(images.cuda(non_blocking=True))#gpu, non_blocking=True) Variable(images.cuda())
        target = Variable(target.cuda(non_blocking=True))#gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    end_time = time.time()
    proc_time = end_time - start_time
    print("Epoch: ", epoch)
    print("Time: ", proc_time)
    print("Loss: ", losses.avg)
    print("Acc1: ", top1.avg.item())
    print("Acc5: ", top5.avg.item())
    train_results[epoch] = {'Loss': losses.avg, 'Acc1': top1.avg.item(), 'Acc5': top5.avg.item(), 'Time': proc_time}


def validate(val_loader, model, criterion, epoch=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        start_time = time.time()
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            #if gpu is not None:
            images = images.cuda(non_blocking=True)#gpu, non_blocking=True)
            target = target.cuda(non_blocking=True)#gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        end_time = time.time()
        proc_time = end_time - start_time
        if epoch is not None:
            val_results[epoch] = {'Loss': losses.avg, 'Acc1': top1.avg.item(), 'Acc5': top5.avg.item(), 'Time': proc_time}

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    new_lr = lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class Unit(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Unit,self).__init__()
        

        self.conv = nn.Conv2d(in_channels=in_channels,kernel_size=3,out_channels=out_channels,stride=1,padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self,input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output

class Net(nn.Module):   
    def __init__(self,num_classes=365):
        super(Net,self).__init__()

        #Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=3,out_channels=64)
        self.unit2 = Unit(in_channels=64, out_channels=64)
        self.unit3 = Unit(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=64, out_channels=128)
        self.unit5 = Unit(in_channels=128, out_channels=128)
        self.unit6 = Unit(in_channels=128, out_channels=128)
        self.unit7 = Unit(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = Unit(in_channels=128, out_channels=256)
        self.unit9 = Unit(in_channels=256, out_channels=256)
        self.unit10 = Unit(in_channels=256, out_channels=256)
        self.unit11 = Unit(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit12 = Unit(in_channels=256, out_channels=512)
        self.unit13 = Unit(in_channels=512, out_channels=512)
        self.unit14 = Unit(in_channels=512, out_channels=512)

        self.avgpool = nn.AvgPool2d(kernel_size=4)
        
        #Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6
                                 ,self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
                                 self.unit12, self.unit13, self.unit14, self.avgpool)

        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        print("Shape: ", output.shape)
        output = output.view(-1,512*7*7)
        output = self.fc(output)
        return output


if __name__ == '__main__':
    main()


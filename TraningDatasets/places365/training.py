from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import torch.utils.data as data
from PIL import Image
import os.path

plt.ion()   # interactive mode

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
		#print(self.root + impath)
		img = self.loader(os.path.join(self.root+impath))
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		
		return img, target

	def __len__(self):
		return len(self.imlist)

def train_model(model, dataloaders, device, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def main():
	# Data augmentation and normalization for training
	# Just normalization for validation
	data_transforms = {
	    'train': transforms.Compose([
	        transforms.RandomResizedCrop(224),
	        transforms.RandomHorizontalFlip(),
	        transforms.ToTensor(),
	        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	    ]),
	    'val': transforms.Compose([
	        transforms.Resize(256),
	        transforms.CenterCrop(224),
	        transforms.ToTensor(),
	        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	    ]),
	}

	data_dir = 'data/places365_Standard/'

	image_datasets = {x: ImageFilelist(root=(data_dir + x), flist=(data_dir + x + ".txt"), 
										transform=data_transforms[x]) 
					for x in ['train', 'val']}

	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
													shuffle=True, num_workers=16) 
					for x in ['train', 'val']}

	dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	model_ft = models.resnet18(num_classes=365)

	model_ft = torch.nn.DataParallel(model_ft).cuda()

	criterion = nn.CrossEntropyLoss()

	# Observe that all parameters are being optimized
	optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

	# Decay LR by a factor of 0.1 every 7 epochs
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

	model_ft = train_model(model_ft, dataloaders, device, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

 # train_loader = torch.utils.data.DataLoader(
 #         ImageFilelist(root="../place365_challenge/data_256/", flist="../place365_challenge/places365_train_challenge.txt",
 #           transform=transforms.Compose([transforms.RandomSizedCrop(224),
 #                transforms.RandomHorizontalFlip(),
 #                transforms.ToTensor(), normalize,
 #       ])),
 #       batch_size=64, shuffle=True,
 #       num_workers=4, pin_memory=True)

 #    val_loader = torch.utils.data.DataLoader(
 #        ImageFilelist(root="../place365_challenge/val_256/", flist="../place365_challenge/places365_val.txt",
 #           transform=transforms.Compose([transforms.Scale(256),
 #               transforms.CenterCrop(224),
 #                transforms.ToTensor(), normalize,
 #        ])),
 #       batch_size=16, shuffle=False,
 #        num_workers=1, pin_memory=True)

	return 0

if __name__ == '__main__':
    main()
from torchvision import models
import torch
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable as V


def load_model():

	model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')

	model.eval()

	return model

def predict(file, model):

	img = Image.open(file)

	transform = transforms.Compose([ 
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225]
		)])

	img_t = transform(img)
	batch_t = img_t.unsqueeze(0)

	input_img = V(batch_t)

	logit = model(input_img)
	h_x = torch.nn.functional.softmax(logit, 1).data.squeeze()*100
	probs, idx = h_x.sort(0, True)
	probs = probs.numpy()
	idx = idx.numpy()
	#out = model(batch_t)

	#print(probs, idx)

	with open('imagenet_synsets.txt', 'r') as f:
		synsets = f.readlines()

	synsets = [x.strip() for x in synsets]
	splits = [line.split(' ') for line in synsets]
	key_to_classname = {spl[0]:' '.join(spl[1:]) for spl in splits}

	with open('imagenet_classes.txt', 'r') as f:
		class_id_to_key = f.readlines()

	class_id_to_key = [x.strip() for x in class_id_to_key]

	counter = 0
	for i in idx[:5]:
		print(key_to_classname[class_id_to_key[i]], probs[counter])
		counter+= 1

if __name__== "__main__":

	model = load_model()

	predict("../dataset/5.jpg", model)


# resnext50 = models.resnet50(pretrained=True)

# transform = transforms.Compose([ 
# 	transforms.Resize(256),
# 	transforms.CenterCrop(224),
# 	transforms.ToTensor(),
# 	transforms.Normalize(
# 	mean=[0.485, 0.456, 0.406],
# 	std=[0.229, 0.224, 0.225]
# 	)])

# img = Image.open("../dataset/0.jpg")

# img_t = transform(img)
# batch_t = torch.unsqueeze(img_t, 0)

# resnext50.eval()

# out = resnext50(batch_t)
# # Load Imagenet Synsets
# with open('imagenet_synsets.txt', 'r') as f:
# 	synsets = f.readlines()

# # len(synsets)==1001
# # sysnets[0] == background
# synsets = [x.strip() for x in synsets]
# splits = [line.split(' ') for line in synsets]
# key_to_classname = {spl[0]:' '.join(spl[1:]) for spl in splits}

# with open('imagenet_classes.txt', 'r') as f:
# 	class_id_to_key = f.readlines()

# class_id_to_key = [x.strip() for x in class_id_to_key]
 
# # Forth, print the top 5 classes predicted by the model
# _, indices = torch.sort(out, descending=True)
# percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
# print([(key_to_classname[class_id_to_key[idx]], percentage[idx].item()) for idx in indices[0][:5]])
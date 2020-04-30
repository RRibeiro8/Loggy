import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import numpy as np
import sys
sys.path.insert(1, 'places365/')
import wideresnet
from tqdm import tqdm
import json

def returnTF():
# load the image transformer
	tf = trn.Compose([
		trn.Resize((224,224)),
		trn.ToTensor(),
		trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	return tf

def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

def load_places365_model():

	model_file = 'places365/wideresnet18_places365.pth.tar'
	
	model = wideresnet.resnet18(num_classes=365)
	checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
	state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
	model.load_state_dict(state_dict)
	model.eval()

	# hook the feature extractor
	features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
	for name in features_names:
		model._modules.get(name).register_forward_hook(hook_feature)
	return model

def load_labels():

	# load the class label
	file_name = 'places365/categories_places365.txt'

	classes = list()
	with open(file_name) as class_file:
		for line in class_file:
			classes.append(line.strip().split(' ')[0][3:])
	classes = tuple(classes)

	# indoor and outdoor relevant
	file_name_IO = 'places365/IO_places365.txt'

	with open(file_name_IO) as f:
		lines = f.readlines()
		labels_IO = []
		for line in lines:
			items = line.rstrip().split()
			labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
	labels_IO = np.array(labels_IO)

	# scene attribute relevant
	file_name_attribute = 'places365/labels_sunattribute.txt'

	with open(file_name_attribute) as f:
		lines = f.readlines()
		labels_attribute = [item.rstrip() for item in lines]
	file_name_W = 'places365/W_sceneattribute_wideresnet18.npy'
	W_attribute = np.load(file_name_W)


	return classes, labels_IO, labels_attribute, W_attribute

def predict_scenes(img_name, model, classes, labels_IO, labels_attribute, W_attribute, tf):

	del features_blobs[:]

	preditions = {}

	img = Image.open(img_name)

	input_img = V(tf(img).unsqueeze(0))

	# forward pass
	logit = model.forward(input_img)
	h_x = F.softmax(logit,1).data.squeeze()
	probs, idx = h_x.sort(0, True)
	probs = probs.numpy()
	idx = idx.numpy()

	io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
	if io_image < 0.5:
		preditions["environment"] = "indoor"
		#print('--TYPE OF ENVIRONMENT: indoor')
	else:
		preditions["environment"] = "outdoor"
		#print('--TYPE OF ENVIRONMENT: outdoor')

	# output the prediction of scene category
	#print('--SCENE CATEGORIES:')
	cat = {}
	for i in range(0, 5):
		#print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
		cat[classes[idx[i]]] = float(probs[i])

	preditions["categories"] = cat
	# output the scene attributes
	responses_attribute = W_attribute.dot(features_blobs[1])
	idx_a = np.argsort(responses_attribute)
	preditions["attributes"] = [labels_attribute[idx_a[i]] for i in range(-1,-10,-1)]
	#print('--SCENE ATTRIBUTES:')
	#print(', '.join([labels_attribute[idx_a[i]] for i in range(-1,-10,-1)]))
	return preditions

if __name__== "__main__":

	classes, labels_IO, labels_attribute, W_attribute = load_labels()
	features_blobs = []
	# load the model
	model = load_places365_model()

	tf = returnTF()

	# load the test image
	#img_name = 'toyshop3.JPG'

	#predict_scenes(img_name, model, classes, labels_IO,labels_attribute, W_attribute, tf)

	DATASET_PATH = "../../Dataset/images/"

	#cat = predict("test2.jpg", model)
	#print(cat)

	for date_dir in tqdm(os.scandir(DATASET_PATH)):

		images_dict = {}

		if date_dir.is_dir():
			print("#################")
			print(date_dir.name)
			
			#if date_dir.name not in processed_dates:
			images_path = os.path.join(date_dir.path)

			for file in tqdm(os.scandir(images_path)):
				if not file.name.startswith(".") and file.is_file():
					#img_fullpath = 
					#print(file.path)
					try:
						prediction = predict_scenes(file.path, model, classes, labels_IO,labels_attribute, W_attribute, tf)

						images_dict[file.name] = prediction
					except:
						print("Image error: ", file.name)

			json_path = "./places365_data/" + date_dir.name + "_places365.json"

			sorted_data = {k: v for k, v in sorted(images_dict.items(), key=lambda item: item[0])}

			with open(json_path, 'w') as jsonfile:
				json.dump(sorted_data, jsonfile, indent=4)
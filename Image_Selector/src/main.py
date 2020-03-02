import numpy as np
import cv2
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import mahotas
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve, ShuffleSplit, StratifiedShuffleSplit, train_test_split
from sklearn import metrics

import pickle

DB_PATH = "dataset/Original/"

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
						n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

	plt.figure()
	plt.title(title)
	if ylim is not None:
		plt.ylim(*ylim)
	plt.xlabel("Training examples")
	plt.ylabel("Score")
	print("Fitting Learning Curves...")
	train_sizes, train_scores, test_scores = learning_curve(
		estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
	print("Done...")
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt.grid()

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
					 train_scores_mean + train_scores_std, alpha=0.1,
					 color="r")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
					 test_scores_mean + test_scores_std, alpha=0.1, color="g")
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
			 label="Training score")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
			 label="Cross-validation score")

	plt.legend(loc="best")
	return plt

def calculate_msv(img):

	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	histogram = cv2.calcHist([img_gray], [0], None, [256], [0,256])

	aux_top = 0
	aux_bottom = 0
	msv = 0

	for j in range(5):
		aux = 0
		for l in range(51):
			aux += histogram[51*j+l]
		aux_top += (j+1)*aux
		aux_bottom += aux
	msv = float(aux_top)/aux_bottom
	#print('Msv: %f' % msv)

	return float(msv) 

def ModifiedLaplacian(img):

	im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#im_norm = cv2.normalize(im_gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

	k = np.array([[-1], [2], [-1]])
	k_t = k.conj().T

	Lx = cv2.filter2D(im_gray, cv2.CV_64F, k, borderType=cv2.BORDER_REPLICATE)
	Ly = cv2.filter2D(im_gray, cv2.CV_64F, k_t, borderType=cv2.BORDER_REPLICATE)

	q = np.abs(Lx) + np.abs(Ly)

	q[np.isnan(q)] = np.min(q)

	return q

# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	feature = cv2.HuMoments(cv2.moments(image)).flatten()
	return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
	# convert the image to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# compute the haralick texture feature vector
	haralick = mahotas.features.haralick(gray).mean(axis=0)
	# return the result
	return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
	bins = 8
	# convert the image to HSV color-space
	image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# compute the color histogram
	hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
	# normalize the histogram
	cv2.normalize(hist, hist)
	# return the histogram
	return hist.flatten()

# feature-descriptor-4: Statistical features from mlap map
def extract_features(q):

	stddev = np.std(q)
	variance = np.var(q)
	weight = np.average(q)

	features = np.array([stddev, variance, weight])

	return features

def main():

	list_dir_blur = sorted(os.listdir(DB_PATH + "Blurred/"))
	list_dir_unblur = sorted(os.listdir(DB_PATH + "Unblurred/"))

	classifier = SVC(C=1, gamma='scale')
	#classifier = KNeighborsClassifier(n_neighbors=4)

	print("Preparing data for training...")

	data_train = []
	labels = []

	for img_name in tqdm(list_dir_blur):

		IMG_PATH = DB_PATH + "Blurred/" + img_name
		img = cv2.imread(IMG_PATH)
		imgr = cv2.resize(img, (512,512))

		q = ModifiedLaplacian(imgr).flatten()
		f3 = extract_features(q)#.flatten()

		data_train.append(f3)
		labels.append(-1)

	for img_name in tqdm(list_dir_unblur):

		IMG_PATH = DB_PATH + "Unblurred/" + img_name
		img = cv2.imread(IMG_PATH)

		imgr = cv2.resize(img, (512,512))

		q = ModifiedLaplacian(imgr).flatten()
		f3 = extract_features(q)#.flatten()

		data_train.append(f3)
		labels.append(1)

	#print(data_train, labels)

	X_train, X_test, y_train, y_test = train_test_split(data_train, labels, test_size=0.25, stratify=labels)

	print("Start SVM training...")
					
	classifier.fit(X_train, y_train)
	print("Training Done!!!")

	train_predict = test_predict = classifier.predict(X_train)

	test_predict = classifier.predict(X_test)
	
	print("Train Classification report for classifier %s:\n%s\n"% (classifier, metrics.classification_report(y_train, train_predict)))
	print("Test Classification report for classifier %s:\n%s\n"% (classifier, metrics.classification_report(y_test, test_predict)))
	print("Confusion Matrix: ", metrics.confusion_matrix(y_test, test_predict))

	print("F1-Score: ", metrics.f1_score(y_test, test_predict))
	#cv = StratifiedShuffleSplit(n_splits=20, test_size=0.3)
	#plot_learning_curve(classifier, "SVM", data_train, labels, ylim=(0.5, 1.01), cv=cv, n_jobs=4)
	#plt.show()

	save_classifier = open("MLAP_SVM.pickle","wb")
	pickle.dump(classifier, save_classifier)
	save_classifier.close()

		#f0 = fd_hu_moments(img)
		#print("feature 0: " , f0)
		#f1 = fd_haralick(img)
		#print("feature 1: " , f1)
		#f2 = fd_histogram(img)
		#print("feature 2: " , f2)

		#cv2.imshow("Image", q)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()

	return 0

if __name__ == "__main__":
	main()
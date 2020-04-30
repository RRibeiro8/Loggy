# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

import os
from tqdm import tqdm
import json

def load_model():
	# Create config
	cfg = get_cfg()
	cfg.merge_from_file("./detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
	cfg.MODEL.WEIGHTS = "./weights/model_final_68b088.pkl"#"detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"

	# Create predictor
	predictor = DefaultPredictor(cfg)

	metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

	return predictor, metadata

def predict(im, predictor, metadata):

	# Make prediction

	preditions = []
	outputs = predictor(im)

	classes = metadata.thing_classes

	idx = outputs["instances"].pred_classes.tolist()
	boxes = outputs["instances"].pred_boxes.tensor.tolist()
	scores = outputs["instances"].scores.tolist()

	for i in range(len(idx)):
		preditions.append({classes[idx[i]]: {"score": scores[i], "box": boxes[i]}})

	#v = Visualizer(im[:, :, ::-1], metadata, scale=1.2)
	#v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
	#cv2.imshow("",v.get_image()[:, :, ::-1])

	#cv2.waitKey()
	return preditions

if __name__== "__main__":

	# get results for image
	#im = cv2.imread("img_test.jpg")
	predictor, metadata = load_model()
	#print(predict(im, predictor, metadata))

	DATASET_PATH = "../../Dataset/images/"

	# processed_dates = ["2015-02-25", "2015-02-26", "2015-02-27", "2015-03-01", "2015-03-05", "2015-03-10", 
	# "2015-03-17", "2015-03-19", "2016-08-11", "2016-08-12", "2016-08-13", "2016-08-17", "2016-08-19", "2016-08-28", 
	# "2016-09-02", "2016-09-04", "2016-09-07", "2016-09-08", "2016-09-13", "2016-09-15", "2016-09-16", "2016-09-17", "2016-09-19",
	# "2016-09-20", "2016-09-24", "2016-09-28", "2016-09-29", "2016-09-30", "2016-10-04", "2018-05-03", "2018-05-05",
	# "2018-05-07", "2018-05-14", "2018-05-24", "2018-05-25", "2018-05-28", "2018-05-31"]
	
	for date_dir in tqdm(os.scandir(DATASET_PATH)):

		images_dict = {}

		if date_dir.is_dir():
			print(date_dir.name)
			
			#if date_dir.name not in processed_dates:
			images_path = os.path.join(date_dir.path)

			for file in tqdm(os.scandir(images_path)):
				if not file.name.startswith(".") and file.is_file():
					#img_fullpath = 
					#print(file.path)
					try:
						im = cv2.imread(file.path)
						concepts = predict(im, predictor, metadata)

						images_dict[file.name] = {"concepts": concepts}
					except:
						print("Image error: ", file.name)

			json_path = "./concepts_data/" + date_dir.name + "_concepts.json"

			sorted_data = {k: v for k, v in sorted(images_dict.items(), key=lambda item: item[0])}

			with open(json_path, 'w') as jsonfile:
				json.dump(sorted_data, jsonfile, indent=4)





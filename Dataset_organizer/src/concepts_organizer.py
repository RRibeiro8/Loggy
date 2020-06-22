import os
from tqdm import tqdm
import json

if __name__== "__main__":

	#with open(os.path.join(settings.MEDIA_ROOT, 'data.json'), 'r') as f:
		#images_info = json.load(f)
		
	concepts_path = os.path.join("../../ImageRecognition/COCO/concepts_data/")

	concepts_files = sorted(os.listdir(concepts_path))
	
	all_concepts_data = {}
	for file_name in concepts_files:

		if file_name.endswith(".json"):
			with open(os.path.join(concepts_path, file_name), 'r') as f:
				images_info = json.load(f)

				print("Json file: ", file_name)

				for img in images_info:

					#print(images_info[img])
					all_concepts_data[img] = images_info[img]


	print(len(all_concepts_data))

	with open('updated_concepts.json', 'w') as jsonfile:
		json.dump(all_concepts_data, jsonfile, indent=4)

import csv
import json
from tqdm import tqdm

def main():

	visual_concepts_file = "../../Dataset/imageclef2020_visual_concepts.csv"
	metadata_file = "../../Dataset/imageclef2020-metadata.csv"

	all_data = {}
	with open(visual_concepts_file, 'r') as csvfile:
		file = csv.reader(csvfile)
		i = 0
		for line in tqdm(file):
			data = {}
			if i > 1:
				#print(line)
				data['minute_id'] = line[0]
				data['utc_time'] = line[1]

				tmp = line[2].split('/')
				image_id = tmp[3]

				data['atributtes'] = line[3:13]

				tmp = line[13:23]

				categories = {}
				for j in range(0,len(tmp),2):
					categories[tmp[j]] = float(tmp[j+1])
				
				data['categories'] = categories
				tmp = line[23:]
				
				concepts = {}
				for j in range(0, len(tmp), 3):
					cont  = {}

					if (tmp[j] != "NULL"):
						box = tmp[j+2].split(' ')
						cont['score'] = float(tmp[j+1])
						cont['box'] = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]

						concepts[tmp[j]] = cont

				data['concepts'] = concepts
				#print(minute_id, utc_time, image_id, atributtes, categories, concepts)

				all_data[image_id] = data
			i += 1

	with open(metadata_file, 'r') as csvfile:
		file = csv.reader(csvfile)
		i = 0
		for line in tqdm(file):
			data = {}
			if i > 1:
				#print(line[0])
				for img_id in all_data:
					#print(all_data[img_id]['minute_id'])
					if line[0] == all_data[img_id]['minute_id']:

						all_data[img_id]['local_time'] = line[2]
						all_data[img_id]['timezone'] = line[3]
						if line[4] == 'NULL':
							all_data[img_id]['latitude'] = None
						else:
							all_data[img_id]['latitude'] = float(line[4])

						if line[4] == 'NULL':
							all_data[img_id]['longitude'] = None
						else:
							all_data[img_id]['longitude'] = float(line[5])
						
						all_data[img_id]['activity'] = line[11]
						all_data[img_id]['location'] = line[6]
						all_data[img_id]['elevation'] = line[7]
						all_data[img_id]['speed'] = line[8]
						all_data[img_id]['heart'] = line[9]
						all_data[img_id]['calories'] = line[10]
						all_data[img_id]['steps'] = line[12]
			
			i += 1

	with open('data.json', 'w') as jsonfile:
		json.dump(all_data, jsonfile, indent=4)

	return 0

if __name__ == "__main__":
	main()
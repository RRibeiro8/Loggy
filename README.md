# Loggy

## General Requirements

- python >= 3.6
- pip
- virtualenv

### Virtual environment

Usually a virtual environment is used for each program or application of this project. In order to create a virtual environment the following commands can be used inside of app folders:

1. virtualenv .
2. source bin/activate

## Dataset

The dataset folder contains the personal lifelog data collection of the ImageCLEF2020 lifelog LMRT task, a rich multimodal dataset focused on daily living activities and the chronological order of the moments, avaliable in:
 > https://www.imageclef.org/2020/lifelog 

The dataset is composed as follows:
- images folder divided into dates
- imageclef2020-metadata.csv
- imageclef2020_visual_concepts.csv
- ImageCLEF2020_dev_clusters.txt
- ImageCLEF2020_dev_gt.txt
- ImageCLEF2020_dev_topics.pdf

## WebApp

This folder contains a web application developed in Django for visualization, summarization and retrieval of moments in the presented dataset.
In order to run the web server locally, a virtual environment is used as mentioned in section above.

### Requeriments

The requirements can be installed using the following command:

> pip install -r requirements.txt

After installing the requirements, it is necessary to download the next models:

> python -m spacy download en_core_web_md

### Database (MariaDB)

In this application, the MariaDB is used. In order to create a new database for Loggy WebApp follow the next steps:

```bash
# Install and perform the necessary initial configuration in you system. Install the packages from the repositories by typing:

sudo apt-get update
sudo apt-get install mariadb-server libmariadbclient-dev libssl-dev

sudo mysql_secure_installation


# Create the database for Loggy WebApp:

sudo mysql -u root -p

CREATE DATABASE Loggy CHARACTER SET UTF8;
CREATE USER admin@localhost IDENTIFIED BY 'admin';
GRANT ALL PRIVILEGES ON Loggy.* TO admin@localhost;
FLUSH PRIVILEGES;
exit

# if you want to delete the database and create another database use:
DROP DATABASE Loggy; 

# After deteting or "drop", create the database following the mentioned commands

```

### Running the Loggy WebApp

Inside of the virtualenviroment the migration have to be done as follows:

```bash
python manage.py makemigrations
python manage.py migrate
```

After the migrations run the web application locally by:

```bash
python manage.py runserver
```

If you want to access to the admin view of Django (/admin/), it needed to create a superuser account as follows:

```bash
python manage.py createsuperuser
```

## Dataset_organizer

This folder contains a program, "main.py" that organize the metadata and visual concepts data from Dataset folder into a json file. The generated json file e organized as follows:

```json
"b00000001_21i6bq_20150223_070808e.jpg": {
        "minute_id": "20150223_0708",
        "utc_time": "UTC_2015-02-23_07:08",
        "atributtes": [
            "no horizon",
            "enclosed area",
            "man-made",
            "glass",
            "indoor lighting",
            "wood",
            "glossy",
            "natural light",
            "matte",
            "cloth"
        ],
        "categories": {
            "wet_bar": 0.057,
            "alcove": 0.046,
            "church/indoor": 0.045,
            "utility_room": 0.042,
            "shower": 0.037
        },
        "concepts": {
            "bottle": {
                "score": 0.987764418,
                "box": [
                    595.58203125,
                    448.6576874788142,
                    618.7140239197531,
                    511.7453828921988
                ]
            },
            "sink": {
                "score": 0.777268946,
                "box": [
                    856.0205439814815,
                    511.2142625919058,
                    1014.1625192901234,
                    568.4875147795874
                ]
            }
        },
```

### optional requirements

> pip install tqdm

## Image_selector

```bibtex
@inproceedings{ribeiro2020image,
  title={Image selection based on low level properties for lifelog moment retrieval},
  author={Ribeiro, Ricardo F and Neves, Ant{\'o}nio JR and Oliveira, Jos{\'e} Luis},
  booktitle={Twelfth International Conference on Machine Vision (ICMV 2019)},
  volume={11433},
  pages={1143303},
  year={2020},
  organization={International Society for Optics and Photonics}
}
```
## ImageRecognition

### COCO

> https://github.com/facebookresearch/detectron2

> https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md

 - faster_rcnn_X_101_32x8d_FPN_3x.yaml
 - model_final_68b088.pkl


### ImageNet


> https://github.com/facebookresearch/FixRes

 - resnext101_32x48d


### Places365

> https://github.com/CSAILVision/places365

 - resnet18




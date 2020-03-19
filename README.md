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

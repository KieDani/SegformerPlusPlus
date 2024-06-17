# Script that can be used to convert the JBD dataset to the coco annotation format
# Required: extracted frames, csv (train, val, bbox)
# --------------------------------------------------------
import json
import os
import shutil

import pandas as pd
from PIL import Image

# folder that contains both train.csv and val.csv
keypoints_path = 'path/to/annotation/directory'
# folder that contains the frames extracted from the videos organized in subdirectories (0-25, one folder per video)
frame_path = 'path/to/frame/directory'
# csv file that contains the bounding box information
bbox_path = 'path/to/bbox.csv'
# out_dir (e.g. mmpose/data/)
out_dir = 'path/to/output/directory'

working_directory = os.getcwd()

# Create directory structure
annotation_directory = os.path.join(out_dir, 'jump_broadcast', 'annotations')
image_directory = os.path.join(out_dir, 'jump_broadcast', 'images')
os.makedirs(annotation_directory, exist_ok=True)
os.chdir(annotation_directory)

os.chdir(working_directory)
os.makedirs(image_directory, exist_ok=True)
os.chdir(image_directory)

os.makedirs('val', exist_ok=True)
os.chdir('val')
for i in range(26):
    os.makedirs(str(i), exist_ok=True)
os.chdir(image_directory)
os.makedirs('train', exist_ok=True)
os.chdir('train')
for i in range(26):
    os.makedirs(str(i), exist_ok=True)
os.chdir(working_directory)


def get_image_size(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


def convert_data_split(annotation_path, bbox_path, image_path, split):
    coco_annotation = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "person", "keypoints": [
            "head", "neck", "rsho", "relb", "rwri", "rhan", "lsho", "lelb", "lwri", "lhan",
            "rhip", "rkne", "rank", "rhee", "rtoe", "lhip", "lkne", "lank", "lhee", "ltoe"],
                        "skeleton": [
                            [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [2, 7], [7, 8], [8, 9], [9, 10],
                            [3, 11], [11, 12], [12, 13], [13, 14], [14, 15],
                            [7, 16], [16, 17], [17, 18], [18, 19], [19, 20]
                        ]
                        }]
    }

    annotation_df = pd.read_csv(annotation_path, skiprows=1, sep=';')
    bbox_df = pd.read_csv(bbox_path, sep=',')
    # remove leading spaces
    bbox_df.columns = bbox_df.columns.str.strip()

    for idx, row in annotation_df.iterrows():
        event = row['event']
        frame_num = str(row['frame_num']).zfill(5)
        image_id = str(event) + '_(' + frame_num + ')'
        image_file = str(event) + '/' + image_id + '.jpg'
        pose_image_id = int(str(event) + frame_num)
        width, height = get_image_size(os.path.join(image_path, image_file))
        image_info = {
            'id': pose_image_id,
            'file_name': image_file,
            'width': width,
            'height': height
        }

        keypoints = []
        for keypoint_name in coco_annotation['categories'][0]['keypoints']:
            x = row[f"{keypoint_name}_x"]
            y = row[f"{keypoint_name}_y"]
            s = row[f"{keypoint_name}_s"]
            keypoints.extend([x, y, s])

        num_keypoints = sum(1 for j in range(0, len(keypoints), 3) if keypoints[j + 2] != 0)
        bbox = bbox_df[bbox_df['image_id'] == image_id]

        # TODO: create own bounding box
        if bbox.empty:
            continue

        bbox = bbox.iloc[0]
        bbox_x = int(bbox['min_x'])
        bbox_y = int(bbox['min_y'])
        bbox_w = int(bbox['width'])
        bbox_h = int(bbox['height'])
        bbox = [bbox_x, bbox_y, bbox_w, bbox_h]
        annotation = {
            'image_id': pose_image_id,
            'id': pose_image_id,
            'category_id': 1,
            'keypoints': keypoints,
            'iscrowd': 0,
            'num_keypoints': num_keypoints,
            'bbox': bbox
        }

        coco_annotation['images'].append(image_info)
        coco_annotation['annotations'].append(annotation)
        print('Added annotation for image', image_id)
        print('Annotation:', annotation)

        src_file = image_path + '/' + image_file
        dst_file = image_directory + '/' + split + '/' + image_file

        shutil.copyfile(src_file, dst_file)
        print('Copied file:', image_id)

    with open(annotation_directory + '/' + split + '_jump.json', 'w') as json_file:
        json.dump(coco_annotation, json_file)


convert_data_split(
    os.path.join(keypoints_path, 'val.csv'),
    bbox_path,
    frame_path,
    'val')
convert_data_split(
    os.path.join(keypoints_path, 'train.csv'),
    bbox_path,
    frame_path,
    'train')

import json
from collections import defaultdict

''' Data pre-processing code used in training.py to parse image annotations in coco format [x_min, y_min, width, height] for bounding boxes. This was developed off a coco training set with images and json annotations.'''

def split_annotations_dict(annotations_dict):
    file_names = []
    annotations = []

    for fname, annots in annotations_dict.items():
        file_names.append(fname)
        annotations.append(annots)

    return file_names, annotations

def parse_annotations(annotation_file, base_path):
    # Load COCO annotations
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    
    # Map image_id to file_name
    id_to_filename = {img['id']: f"{base_path}{img['file_name']}" for img in images}

    # Initialize a dictionary to hold lists of [bbox + category_id]
    grouped_annotations = defaultdict(list)

    # Group annotations by image file name
    for ann in annotations:
        image_id = ann['image_id']
        file_name = id_to_filename.get(image_id, None)
        if file_name:
            bbox = ann['bbox']  # [x, y, width, height]
            category_id = ann['category_id']
            bbox_with_cat = bbox + [category_id]
            grouped_annotations[file_name].append(bbox_with_cat)

    # Optional: convert to regular dict if needed
    grouped_annotations = dict(grouped_annotations)


    file_names, annotations = split_annotations_dict(grouped_annotations)

    return file_names, annotations


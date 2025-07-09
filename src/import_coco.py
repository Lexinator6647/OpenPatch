from pycocotools.coco import COCO
import os, shutil
import requests
import cv2
from tqdm import tqdm
stage = "val"
coco_dir = f"annotations/annotations/instances_{stage}2017.json"

save_dir = f"dog_images/{stage}"
#os.makedirs(coco_dir, exist_ok=True)
coco = COCO(coco_dir)
dog_category_id = 18
# Get all image IDs that contain dogs
img_ids = coco.getImgIds(catIds=[dog_category_id])
print(f"Found {len(img_ids)} images with dogs.")
# Limit to 10 for a quick start
img_ids = img_ids[:50]

# Fetch image metadata
imgs = coco.loadImgs(img_ids)
#print(f"len imgs: {len(imgs)}")
#print(f"tqdm imgs: {tqdm(imgs)}")
#for img in imgs:
for img_info in tqdm(imgs):
    #print(f"img_info: {img_info}")
    file_name = img_info['file_name']
    #print(f"file name to download: {file_name}")
    url = img_info['coco_url']
    #print(f"coco url: {url}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    #print(f"save_path: {save_path}")
    #if os.path.exists(save_dir):
        #continue
    try:
        print(f"trying url {url}")
        response = requests.get(url, stream = True)
        if response.status_code == 200:
            print(f"successful request")
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
        else:
            print(f"Failed to download {file_name} from {url}")

    except Exception as e:
        print(f"Error downloading {file_name}: {e}")

output_dir = f"COCO_dog_subset/{stage}"
os.makedirs(f"{output_dir}/images", exist_ok=True)
os.makedirs(f"{output_dir}/annotations", exist_ok=True)
subset_annotations = {"images": [], "annotations": [], "categories": coco.loadCats(coco.getCatIds())}
for img_id in img_ids:
    img_info = coco.loadImgs(img_id)[0]
    file_name = f"{img_info['file_name']}"
    print(f"file name: {file_name}")
    shutil.copy(f"dog_images/{stage}/{file_name}", f"{output_dir}/images/{file_name}")
    subset_annotations["images"].append(img_info)
    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[dog_category_id], iscrowd=False)
    anns = coco.loadAnns(ann_ids)
    subset_annotations["annotations"].extend(anns)
# Save annotations
import json
with open(f"{output_dir}/annotations/instances_dog_subset.json", "w") as f:
    json.dump(subset_annotations, f)
print("Saved dog subset!")


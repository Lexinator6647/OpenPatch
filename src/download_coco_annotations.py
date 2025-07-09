import os
import zipfile
import urllib.request
def download_and_extract_coco_annotations(dest_dir="annotations"):
    os.makedirs(dest_dir, exist_ok=True)
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    zip_path = os.path.join(dest_dir, "annotations_trainval2017.zip")
    if not os.path.exists(zip_path):
        print("Downloading COCO annotations...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
    else:
        print("Annotations zip already exists. Skipping download.")
    # Extract only if necessary
    expected_file = os.path.join(dest_dir, "annotations", "instances_train2017.json")
    if not os.path.exists(expected_file):
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        print("Extraction complete.")
    else:
        print("Annotations already extracted.")
if __name__ == "__main__":
    download_and_extract_coco_annotations()







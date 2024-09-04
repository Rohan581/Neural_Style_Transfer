import os
from pathlib import Path
from zipfile import ZipFile, BadZipFile
import requests

cwd = Path(__file__).parent.absolute()

coco_path = str(cwd) + '/coco'
zip_url_train = "http://images.cocodataset.org/zips/train2017.zip"
zip_url_val = "http://images.cocodataset.org/zips/val2017.zip"
zip_url_ann = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
zip_file_path_train = os.path.join(coco_path, "coco_train2017.zip")
zip_file_path_val = os.path.join(coco_path, "coco_val2017.zip")
zip_file_path_ann = os.path.join(coco_path, "coco_ann2017.zip")

if not os.path.exists(coco_path):
    print("Downloading coco dataset 2017...")
    os.makedirs(coco_path, exist_ok=True)
    os.system(f"wget {zip_url_train} -O {zip_file_path_train}")
    os.system(f"wget {zip_url_val} -O {zip_file_path_val}")
    os.system(f"wget {zip_url_ann} -O {zip_file_path_ann}")
else:
    print("Folder already exists. Skipping download.")


def extract_zip_file(extract_path):
    try:
        with ZipFile(extract_path + ".zip") as zfile:
            zfile.extractall(extract_path)
        # remove zipfile
        ZfileToRemove = f"{extract_path}" + ".zip"
        if os.path.isfile(ZfileToRemove):
            os.remove(ZfileToRemove)
        else:
            print("Error: %s file not found" % ZfileToRemove)
    except BadZipFile as e:
        print("Error:", e)


if not os.path.exists(coco_path + '/coco_train2017'):
     extract_zip_file(coco_path + '/coco_train2017')

if not os.path.exists(coco_path + '/coco_val2017'):
    extract_zip_file(coco_path + '/coco_val2017')

if not os.path.exists(coco_path + '/coco_ann2017'):
    extract_zip_file(coco_path + '/coco_ann2017')

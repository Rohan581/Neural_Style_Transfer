from train import train
from style_transfer import style_transfer
from pathlib import Path
import os
import time

cwd = Path(__file__).parent.absolute()

style_image_path = str(cwd) + '/style_imgs/Starry_Night.jpg'
dataset = str(cwd) + '/coco/coco_train2017/train2017/'

style_name = os.path.split(style_image_path)[-1].split('.')[0]

if not os.path.exists(str(cwd) + "/models/" + str(style_name) + ".model"):
    train(style_image_path,dataset,visualize=False)
else:
    folder_name = 'Output'

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    model_path = str(cwd) + "/models/" + str(style_name)  + ".model"
    source_image_path = str(cwd) + '/content_imgs/Mim.jpg'
    source_name = os.path.split(source_image_path)[-1].split('.')[0]
    output = str(cwd)+ '/Output/' + str(style_name) + '_' + str(source_name) + '.jpg'
    style_transfer(model_path,source_image_path,style_image_path,output, gpu=0)


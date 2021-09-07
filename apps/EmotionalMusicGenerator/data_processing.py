import os
from random import random
import shutil
from tqdm import tqdm


def check_folder(ff):
    if not os.path.exists(ff):
        os.mkdir(ff)


image_dir = '../dataset/train'
image_classes = os.listdir(image_dir)
target_dir = '../dataset/ready_to_use/'
check_folder(target_dir)
image_sets = ['train', 'valid']
emotion_mapping = {
    'happy': 'Q1',
    'surprise': 'Q1',
    'angry': 'Q2',
    'disgust': 'Q2',
    'fear': 'Q2',
    'sad': 'Q3',
    'neutral': 'Q4',
}


def fer_data():
    for sets in image_sets:
        this_dir = os.path.join(target_dir, sets)
        check_folder(this_dir)
        for ic in image_classes:
            this_ic_dir = os.path.join(this_dir, ic)
            check_folder(this_ic_dir)

    for ic in image_classes:
        this_image_dir = os.path.join(image_dir, ic)
        print('-'*10, ic, '-'*10)
        for images in tqdm(os.listdir(this_image_dir)):
            this_set = 'train' if random() > 0.1 else 'valid'
            this_dir = os.path.join(target_dir, this_set, emotion_mapping[ic])
            check_folder(this_dir)
            shutil.copy2(os.path.join(this_image_dir, images), os.path.join(this_dir, images))
            check_folder(f'C:/dataset/emotion_image/{emotion_mapping[ic]}')

            if random() > 0.95:
                shutil.copy2(os.path.join(this_image_dir, images),
                             f'C:/dataset/emotion_image/{emotion_mapping[ic]}/{emotion_mapping[ic]}_{images}')


fer_data()


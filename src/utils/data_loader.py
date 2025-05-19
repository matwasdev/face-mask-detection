import os

import cv2
import numpy as np

from src.utils import face_extractor


def load_images():
    images_folder = 'D:\\AI_ML\\Face Mask Detection\\masks\\images'
    images_info_folder = 'D:\\AI_ML\\Face Mask Detection\\masks\\annotations'

    images_paths = os.listdir(images_folder)
    images_info_paths = os.listdir(images_info_folder)

    images_with_info_paths = {}

    for i in range(len(images_paths)):
        images_paths[i] = images_folder + "\\" + images_paths[i]
        images_info_paths[i] = images_info_folder + "\\" + images_info_paths[i]

        images_with_info_paths[images_paths[i]] = images_info_paths[i]

    return images_with_info_paths



def get_dataset():
    images = []
    labels = []

    images_with_info_paths = load_images()

    for image, info in images_with_info_paths.items():
        faces_with_labels = face_extractor.extract_faces_and_labels_from_image_xml(image, info)
        for tup in faces_with_labels:
            images.append(tup[0])
            labels.append(tup[1])


    return images, labels


def change_class_occurrences(images,labels,max_occurrences=1200):
    labels_set = set(labels)
    label_occurrences = {label: 0 for label in labels_set}

    new_images = []
    new_labels = []

    for i in range(len(images)):
        if labels[i] in label_occurrences and label_occurrences[labels[i]] < max_occurrences:
            label_occurrences[labels[i]] += 1
            new_images.append(images[i])
            new_labels.append(labels[i])

    print("WITH MASK: "+ str(labels.__len__()))
    print(label_occurrences)

    return new_images, new_labels



def prepare_for_training(images, labels):
    for i in range(len(images)):
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        images[i] = cv2.resize(images[i], (32, 32))

    for i in range(len(labels)):
        if labels[i] == 'with_mask':
            labels[i] = 1
        else:
            labels[i] = 0

    images = np.array(images).astype('float32') / 255.0
    labels = np.array(labels).astype('int32')

    return images, labels



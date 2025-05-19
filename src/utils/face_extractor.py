import xml.etree.ElementTree as ET
import cv2


def extract_faces_and_labels_from_image_xml(image_path,image_xml_path):
    faces_with_labels = []

    xml_parsed = ET.parse(image_xml_path)
    xml_root = xml_parsed.getroot()

    image_objects = xml_root.findall('object')

    image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
    # print(image)

    for i, info in enumerate(image_objects):
        label = info.find('name').text
        if label == "mask_weared_incorrect":
            continue
        bndbox = info.find('bndbox')
        # print("LABEL + " + label)

        x_min = int(bndbox.find('xmin').text)
        y_min = int(bndbox.find('ymin').text)
        x_max = int(bndbox.find('xmax').text)
        y_max = int(bndbox.find('ymax').text)

        face = image[y_min:y_max, x_min:x_max]
        # face = cv2.resize(face, (100, 100))
        faces_with_labels.append( ( face,label ) )

    return faces_with_labels




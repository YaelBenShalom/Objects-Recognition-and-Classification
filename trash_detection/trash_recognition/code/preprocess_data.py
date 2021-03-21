import os
import shutil
import numpy as np
import tqdm
import splitfolders
from pycocotools.coco import COCO

# Define input data file
data_source = COCO(annotation_file='data/data/annotations.json')

# Define output folder
save_base_path = 'data/data/tmp/labels/'
save_image_path = 'data/data/tmp/images/'

# Required labels
#  'Clear plastic bottle': 5
#  'Glass bottle': 6
#  'Drink can': 12
#  'Drink carton': 16
#  'Paper cup': 20
#  'Disposable plastic cup': 21
#  'Foam cup': 22
#  'Normal paper': 33
#  'Single-use carrier bag': 40
#  'Crisp packet': 42

# Remapping label id from 0-59 to 0-9
label_transfer = {5: 0, 6: 1, 12: 2, 16: 3, 20: 4, 21: 5, 22: 6, 33: 7, 40: 8, 42: 9}

img_ids = data_source.getImgIds()

catIds = data_source.getCatIds()
categories = data_source.loadCats(catIds)
categories.sort(key=lambda x: x['id'])
classes = {}
coco_labels = {}
coco_labels_inverse = {}
cd 
for c in categories:
    coco_labels[len(classes)] = c['id']
    coco_labels_inverse[c['id']] = len(classes)
    classes[c['name']] = len(classes)

class_num = {}
print(classes)

# Convert .jason file to .txt file
for index, img_id in tqdm.tqdm(enumerate(img_ids), desc='change .json file to .txt file'):

    # Extract image information
    img_info = data_source.loadImgs(img_id)[0]
    save_name = img_info['file_name'].replace('/', '_')
    file_name = save_name.split('.')[0]
    height = img_info['height']
    width = img_info['width']
    save_path = save_base_path + file_name + '.txt'

    # Degine exist flag
    is_exist = False

    with open(save_path, mode='w') as fp:
        annotation_id = data_source.getAnnIds(img_id)
        # boxes = np.zeros((0, 5))

        if len(annotation_id) == 0:
            fp.write('')
            continue

        # Define image annotations
        annotations = data_source.loadAnns(annotation_id)
        lines = ''
        
        for annotation in annotations:
            label = coco_labels_inverse[annotation['category_id']]
            print("label: ", label)
            if label in label_transfer.keys():
                is_exist = True
                box = annotation['bbox']

                if box[2] < 1 or box[3] < 1:
                    continue

                # convert boxes to (cen_x, cen_y, width, height)
                box[0] = round((box[0] + box[2] / 2) / width, 6)
                box[1] = round((box[1] + box[3] / 2) / height, 6)
                box[2] = round(box[2] / width, 6)
                box[3] = round(box[3] / height, 6)

                # Convert labels to remapped labels
                label = label_transfer[label]
                print("new label: ", label)

                if label not in class_num.keys():
                    class_num[label] = 0
                class_num[label] += 1
                lines = lines + str(label)

                for i in box:
                    lines += ' ' + str(i)
                lines += '\n'

        fp.writelines(lines)

    # Save the output in tmp folder    
    if is_exist:
        shutil.copy('data/data/{}'.format(img_info['file_name']), os.path.join(save_image_path, save_name))
    else:
        os.remove(save_path)

# Split data randomly to train, validation, and test folders (with ratio 0.8, 0.1, 0.1)
splitfolders.ratio('data/data/tmp', output="data/data/taco", seed=1337, ratio=(0.8, 0.1, 0.1))

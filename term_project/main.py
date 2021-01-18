#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
print(tf.__version__)


# In[ ]:


get_ipython().run_line_magic('cd', 'keras-yolo3')


# In[ ]:


import os
if not os.path.exists("model_data/yolo.h5"):
    print("Model doesn't exist, downloading...")
    os.system("wget https://pjreddie.com/media/files/yolov3.weights")
    print("Converting yolov3.weights to yolo.h5...")
    os.system("python3 convert.py yolov3.cfg yolov3.weights model_data/yolo.h5")
else:
    print("Model exist")


# In[ ]:


from glob import glob
imgs_path = glob("/data/NFS/andy_data/yolo/kaggle_facemask/images/*.png")
imgs_path.sort()
num = int(len(imgs_path)*0.1)

f = open("/data/NFS/andy_data/yolo/kaggle_facemask/train.txt", "w")
for i in range(num*9):
    name = imgs_path[i][48:-4]
    f.write(name + "\n")
f.close()

f = open("/data/NFS/andy_data/yolo/kaggle_facemask/val.txt", "w")
for i in range(num*9, len(imgs_path)):
    name = imgs_path[i][48:-4]
    f.write(name + "\n")
f.close()


# In[ ]:


import xml.etree.ElementTree as ET
from os import getcwd

sets=['train', 'val']

# Facemask 的資料類別
classes = ["with_mask", "without_mask", "mask_weared_incorrect"]

# 把 annotation 轉換訓練時需要的資料形態
def convert_annotation(image_id, list_file):
    in_file = open('/data/NFS/andy_data/yolo/kaggle_facemask/annotations/%s.xml'%(image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

for image_set in sets:
    image_ids = open('/data/NFS/andy_data/yolo/kaggle_facemask/%s.txt'%(image_set)).read().strip().split()
    annotation_path = '%s.txt'%(image_set)
    list_file = open(annotation_path, 'w')
    print("save annotation at %s" % annotation_path)
    for image_id in image_ids:
        list_file.write('/data/NFS/andy_data/yolo/kaggle_facemask/images/%s.png'%(image_id))
        convert_annotation(image_id, list_file)
        list_file.write('\n')
    list_file.close()


# In[ ]:


import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data


# In[ ]:


from train import get_classes, get_anchors, create_model, create_tiny_model, data_generator, data_generator_wrapper


# In[ ]:


if not os.path.exists("model_data/yolo_weights.h5"):
    print("Converting pretrained YOLOv3 weights for training")
    os.system("python3 convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5") 
else:
    print("Pretrained weights exists")


# In[ ]:


import warnings
warnings.filterwarnings("ignore")

annotation_path = 'train.txt' # 轉換好格式的標註檔案
log_dir = 'logs/000/' # 訓練好的模型儲存的路徑
classes_path = 'model_data/voc_classes.txt'
anchors_path = 'model_data/yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)

input_shape = (416,416)

is_tiny_version = len(anchors)==6 
if is_tiny_version:
    model = create_tiny_model(input_shape, anchors, num_classes,
        freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
else:
    model = create_model(input_shape, anchors, num_classes,
        freeze_body=2, weights_path='model_data/yolo_weights.h5') 

logging = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
    monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)


val_split = 0.1
with open(annotation_path) as f:
    lines = f.readlines()
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
num_val = int(len(lines)*val_split)
num_train = len(lines) - num_val

# Train with frozen layers first, to get a stable loss.
# Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
# 一開始先 freeze YOLO 除了 output layer 以外的 darknet53 backbone 來 train
if True:
    model.compile(optimizer=Adam(lr=1e-3), loss={
        'yolo_loss': lambda y_true, y_pred: y_pred})

    batch_size = 64
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    history1 = model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=50,
            initial_epoch=0,
            callbacks=[logging, checkpoint])
    model.save_weights(log_dir + 'trained_weights_stage_1.h5')

# Unfreeze and continue training, to fine-tune.
if True:
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) 
    print('Unfreeze all of the layers.')

    batch_size = 4 
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    history2 = model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
        steps_per_epoch=max(1, num_train//batch_size),
        validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
        validation_steps=max(1, num_val//batch_size),
        epochs=100,
        initial_epoch=50,
        callbacks=[logging, checkpoint, reduce_lr, early_stopping])
    model.save_weights(log_dir + 'trained_weights_final.h5')


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(history1.history['loss'] + history2.history['loss'])
plt.plot(history1.history['val_loss'] + history2.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()


# In[ ]:


annotation_path = 'train.txt' # 轉換好格式的標註檔案
log_dir = 'logs/000/' # 訓練好的模型儲存的路徑
classes_path = 'model_data/voc_classes.txt'
anchors_path = 'model_data/yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)

from yolo import YOLO
yolo_model = YOLO(model_path=log_dir + 'trained_weights_final.h5', classes_path=classes_path)


# In[ ]:


from PIL import Image
image = Image.open("/data/NFS/andy_data/yolo/kaggle_facemask/images/maksssksksss805.png")
r_image = yolo_model.detect_image(image)
r_image


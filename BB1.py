#training with data augmentation and batch_size=32 is the best.
#we modified the get_bbox according to our annotations
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import glob
import PIL
import cv2
import xmltodict
import random
from tqdm import tqdm
from PIL import ImageDraw
import tensorflow as tf
from tensorflow.keras.losses import Loss
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPool2D, LeakyReLU, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import xml.etree.ElementTree as ET

path = "training_images"
print(path)
def normalize(path):
    images=[]
    names =[]
    for file in tqdm(glob.glob(path +"/*.jpg")):
        image = cv2.resize(cv2.imread(file), (228,228))
        image = np.array(image)
        name = file.split('/')[-1].split('_')[0]
        images.append(image)
        names.append(name)
    return images,names
images,names = normalize(path)

fig = plt.figure(figsize=(20,15))

for i in range(9):
    r = random.randint(1,186)
    plt.subplot(3,3,i+1)
    plt.imshow(images[r])
    plt.xlabel(names[r])
    
plt.show()

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)




def get_bbox(xml_path):
    bboxes = []
    classnames = []
    for file in tqdm(glob.glob(xml_path + "/*.xml")):
        tree = ET.parse(file)
        root = tree.getroot()
        img_width = int(root.find('size').find('width').text)
        img_height = int(root.find('size').find('height').text)
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            name = obj.find('name').text
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            # Normalize bounding box coordinates
            bbox = np.array([xmin / img_width, ymin / img_height, xmax / img_width, ymax / img_height])
            bboxes.append(bbox)
            classnames.append(name)
    return np.array(bboxes), classnames

bboxes, classnames = get_bbox(path)

encoder = LabelBinarizer()
classnames = encoder.fit_transform(classnames)
Y = np.concatenate([bboxes, classnames], axis=1)
X = np.array(images)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1)

train_datagen = datagen.flow(X_train, Y_train, batch_size=15)


print('X_train:', X_train.shape, 'Y_train:', Y_train.shape, 'X_test:', X_test.shape, 'Y_test:', Y_test.shape)

def calculate_iou(target, pred):
    target = tf.cast(target, tf.float32)
    pred = tf.cast(pred, tf.float32)
    
    xA = K.maximum(target[:,0], pred[:,0])
    yA = K.maximum(target[:,1], pred[:,1])
    xB = K.minimum(target[:,2], pred[:,2])
    yB = K.minimum(target[:,3], pred[:,3])
    interArea = K.maximum(0.0, xB-xA) * K.maximum(0.0, yB-yA)
    boxAarea = (target[:,2]-target[:,0]) * (target[:,3]-target[:,1])
    boxBarea = (pred[:,2]-pred[:,0]) * (pred[:,3]-pred[:,1])
    
    iou = interArea / (boxAarea + boxBarea - interArea)
    return iou

class MeanSquaredError(Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true), axis=-1)

def custom_loss(y_true, y_pred):
    mse = MeanSquaredError()
    mse_loss = mse(y_true, y_pred)
    iou = calculate_iou(y_true , y_pred)
    return mse_loss + (1 - iou)

def iou_metric(y_true, y_pred):
    return calculate_iou(y_true, y_pred)

input_shape = ( 228 , 228 , 3 )
dropout_rate = 0.5
classes =1
alpha = 0.2
prediction_units = 4+ classes



def block1(filters,X):
    
    x = Conv2D(filters, kernel_size=(3,3), strides=1 )(X)
    x = LeakyReLU(alpha)(x)
    x = Conv2D(filters, (3,3), strides=1)(x)
    x = LeakyReLU(alpha)(x)
    x = MaxPool2D((2,2))(x)
    return x

def block2(units,X):
    
    x = Dense(units)(X)
    x = LeakyReLU(alpha)(x)    
    return x

def mymodel():
    
    model_input = Input(shape=(228,228,3))
    x= block1(16, model_input)
    x= block1(32,x)
    x = block1(64,x)
    x= block1(128,x)
    x = block1(256,x)
    
    x = Flatten()(x)
    x = block2(1240, x)
    x = block2(640, x)
    x = block2(480,x)
    x = block2(120,x)
    x = block2(62,x)
    model_outputs = Dense(prediction_units)(x)
    
    model = Model(inputs=[model_input], outputs=[model_outputs])
    
    model.compile( tf.keras.optimizers.Adam(0.0001),
                 loss=custom_loss,
                 metrics=[iou_metric])
    return model

model = mymodel()
model.summary()
#plot_model(model,to_file='model.png',show_shapes=True,show_layer_names=True)

from tensorflow.keras.callbacks import ModelCheckpoint

filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.weights.h5'
checkpoint = ModelCheckpoint(filepath,save_weights_only=True, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = model.fit(
    train_datagen,
    validation_data=(X_test, Y_test),
    epochs=200,
    steps_per_epoch=len(X_train) // 15,  # Number of batches per epoch
    callbacks=[checkpoint]
)


loss = history.history['loss']
val_loss = history.history['val_loss']

acc = history.history['iou_metric']
val_acc = history.history['val_iou_metric']


plt.figure(figsize=(10,15))
plt.subplot(2,1,1)
plt.plot(loss , linewidth=3 ,label='train loss')
plt.plot(val_loss , linewidth=3, label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss / val_loss')
plt.legend()
plt.show()

plt.subplot(2,1,2)
plt.plot(acc , linewidth=3 ,label='train acc')
plt.plot(val_acc , linewidth=3, label='val acc')
plt.xlabel('epochs')
plt.ylabel('Accuracy / Val_Accuracy')
plt.legend()
plt.show()

def drawbox(model,image, y_true, le):
    img = tf.cast(np.expand_dims(image, axis=0), tf.float32)
    y_true = np.expand_dims(y_true, axis=0)
    
    #prediction
    predict = model.predict(img)
    
    #Box coordinates
    Y_test_box =y_true[...,0:4]*228
    pred_box = predict[...,0:4]*228
    
    x = pred_box[0][0]
    y = pred_box[0][1]
    w = pred_box[0][2]
    h = pred_box[0][3]
    #get class name
    trans= le.inverse_transform(predict[...,4:])
    
    im = PIL.Image.fromarray(image)
    draw=ImageDraw.Draw(im)
    draw.rectangle([x,y,w,h], outline='red')
    plt.xlabel(trans[0])
    plt.imshow(im)
    
    iou = calculate_iou(Y_test_box, pred_box)

fig = plt.figure(figsize=(20,15))

for i in range(9):
    r = random.randint(1,10)
    plt.subplot(3,3,i+1)
    drawbox(model,X_test[r], Y_test[r], encoder)    
plt.show()


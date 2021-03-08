#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


(x_train,y_train),(x_test,y_test)=keras.datasets.cifar10.load_data()


# In[3]:


x_train.shape


# In[4]:


classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# In[5]:


def show(x,y,index):
    plt.figure(figsize=(15,2))
    plt.imshow(x[index])
    plt.xlabel(classes[y[index][0]],color="blue")


# In[6]:


show(x_train,y_train,2000)


# In[7]:


x_train=x_train/255
x_test=x_test/255


# In[22]:


Ann_model=keras.Sequential([
    keras.layers.Flatten(input_shape=(32,32,3)),
    keras.layers.Dense(700,activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(300,activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(170,activation="relu"),
    keras.layers.Dense(10,activation="sigmoid")
])
Ann_model.compile(optimizer="SGD",loss="sparse_categorical_crossentropy",metrics=['accuracy'])
Ann_model.fit(x_train,y_train,epochs=50)


# In[23]:


Ann_model.evaluate(x_train,y_train)


# In[24]:


Ann_model.evaluate(x_test,y_test)


# In[16]:


from sklearn.metrics import classification_report
y_pred=Ann_model.predict(x_test)
y_pred_classes=[np.argmax(element) for element in y_pred]
print("classification_report: \n",classification_report(y_test,y_pred_classes))


# In[34]:


Cnn_model=keras.Sequential([
    #cnn
    keras.layers.Conv2D(24,(3,3),activation="relu",input_shape=(32,32,3)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(45,(3,3),activation="relu"),
    keras.layers.MaxPooling2D((2,2)),
    #dense_layers
    keras.layers.Flatten(),
    keras.layers.Dense(1500,activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10,activation="softmax"),
])
Cnn_model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])
Cnn_model.fit(x_train,y_train,epochs=50)


# In[35]:


Cnn_model.evaluate(x_train,y_train)


# In[36]:


Cnn_model.evaluate(x_test,y_test)


# In[55]:


Ann_model.evaluate(x_train,y_train)


# In[56]:


Ann_model.evaluate(x_test,y_test)


# In[37]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
cnn_model=keras.Sequential([
    #cnn
    keras.layers.Conv2D(24,(3,3),activation="relu",input_shape=(32,32,3)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(45,(3,3),activation="relu"),
    keras.layers.MaxPooling2D((2,2)),
    #dense_layers
    keras.layers.Flatten(),
    keras.layers.Dense(1500,activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10,activation="softmax"),
])
cnn_model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])


# In[ ]:


cnn_model.fit(datagen.flow(x_train, y_train, batch_size=64),epochs=50,steps_per_epoch = x_train.shape[0] // 100)


# In[31]:


x_train.shape[0]


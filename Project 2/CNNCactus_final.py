import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os		#,cv2
from keras import optimizers
from keras import layers,models
from keras.preprocessing.image import ImageDataGenerator
print(os.listdir("aci/"))
from keras.models import load_model

import numpy as np

train_dir="aci/train"

train=pd.read_csv('aci/train.csv')

train.head(5)
train.has_cactus=train.has_cactus.astype(str)
print('out dataset has {} rows and {} columns'.format(train.shape[0],train.shape[1]))
datagen=ImageDataGenerator(rescale=1./255)
batch_size=150
train_generator=datagen.flow_from_dataframe(dataframe=train[:12000],directory=train_dir,x_col='id',y_col='has_cactus',class_mode='binary',batch_size=batch_size,target_size=(150,150))
validation_generator=datagen.flow_from_dataframe(dataframe=train[12000:15000],directory=train_dir,x_col='id',y_col='has_cactus',class_mode='binary',batch_size=50,target_size=(150,150))
test_generator=datagen.flow_from_dataframe(dataframe=train[15000:],directory=train_dir,x_col='id',y_col='has_cactus',class_mode='binary',batch_size=batch_size,target_size=(150,150))

t1=train[15000:]
t1_1 = t1[t1['has_cactus']=='1']
t1_0 = t1[t1['has_cactus']=='0']

t1_1.at[15000,'has_cactus']='0'
t1_0.at[15003,'has_cactus']='1'

test_generator_1=datagen.flow_from_dataframe(dataframe=t1_1,directory=train_dir,x_col='id',y_col='has_cactus',class_mode='binary',batch_size=batch_size,target_size=(150,150))
test_generator_0=datagen.flow_from_dataframe(dataframe=t1_0,directory=train_dir,x_col='id',y_col='has_cactus',class_mode='binary',batch_size=batch_size,target_size=(150,150))


print(test_generator.samples)
print(validation_generator.samples)
print(train_generator.samples)

model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer=optimizers.rmsprop(),metrics=['acc'])
epochs=10

#history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=10,validation_data=validation_generator,validation_steps=50)
#model.save('my_model_01.hdf5')

model = load_model('my_model_01.hdf5')

score=model.evaluate_generator(test_generator , steps=test_generator.samples//batch_size+1)
print(model.metrics_names)
print('Score : ',score)

score_1=model.evaluate_generator(test_generator_1 , steps=test_generator_1.samples//batch_size+1)
print('Score 1 : ',score_1)

score_0=model.evaluate_generator(test_generator_0 , steps=test_generator_0.samples//batch_size+1)
print('Score 0 : ',score_0)



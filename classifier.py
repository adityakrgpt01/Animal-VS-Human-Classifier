#Importing Libraries
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

#Importing Inception_V3 trained on Imagenet
from keras.applications.inception_v3 import InceptionV3
base_model = InceptionV3(weights='imagenet', include_top=True)
base_model.summary()

#Removing last layer and adding some dense layer and Prediction layer of 2 classes
x= base_model.layers[-2].output
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(2,activation='softmax')(x)

from keras.models import Model

model_final = Model(input = base_model.input, output = preds)

model_final.summary()

for i,layer in enumerate(model_final.layers):
  print(i,layer.name)
  
for layers in (model_final.layers)[:312]:
    print(layers)
    layers.trainable = False
    
from keras.optimizers import Adam
opt = Adam(lr=0.001)

model_final.compile(optimizer=opt,loss="binary_crossentropy",metrics=["accuracy"])


#Preprocessing Image Dataset
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory(
        'huma_Animal_dataset/train',
        target_size=(299, 299),
        batch_size=40)

test_set = test_datagen.flow_from_directory(
        'huma_Animal_dataset/test',
        target_size=(299,299),
        batch_size=40)


#Saving The best validation accuracy model
from keras.callbacks import ModelCheckpoint, EarlyStopping
path = F"classifier_model.h5"
checkpoint = ModelCheckpoint(path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

#Handling truncated_images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#Training the model
model_final.fit_generator(
        training_set,
        steps_per_epoch=195,
        epochs=50,
        validation_data=test_set,
        validation_steps=37,callbacks=[checkpoint])
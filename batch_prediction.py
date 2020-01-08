#Prediction on Images of a folder

#Imorting Libraries
import numpy as np
from keras.preprocessing import image
from numpy import asarray
import os
from keras.models import load_model
import matplotlib.pyplot as plt

#Loading model and compiling it
model_final = load_model("classifier_model.h5")
from keras.optimizers import Adam
opt = Adam(lr=0.001)
model_final.compile(optimizer=opt,loss="binary_crossentropy",metrics=["accuracy"])



#Preposing Images of a folder and Predicting result
folder_path='Sample_Input'
for img in os.listdir(folder_path):
    img = os.path.join(folder_path, img)
    # img = rescale(image, 1/255, anti_aliasing=False)
    
    img = image.load_img(img, target_size=(299, 299))
    plt.imshow(img)
    plt.show()
    img=  asarray(img).astype('float32')/255
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    classes = model_final.predict(img)
    # print(classes)
    if classes[0][0]>classes[0][1]:
        print("animal")
        
    else:
        print("human")
        

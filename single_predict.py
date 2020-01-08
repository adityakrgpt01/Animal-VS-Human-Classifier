# make a single prediction
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from keras.preprocessing import image
from keras.models import load_model

#taking input image and preprossesing it
test_image=image.load_img("Sample_Input/0-3667_young-people-dance-png-transparent-png.jpg",target_size=(299,299))
plt.imshow(test_image)
plt.show()
test_image=  asarray(test_image).astype('float32')/255
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)


#loading the trained model and compiling it
saved_model = load_model("classifier_model.h5")
from keras.optimizers import Adam
opt = Adam(lr=0.001)
saved_model.compile(optimizer=opt,loss="binary_crossentropy",metrics=["accuracy"])


#Prediction of Load Image
result=saved_model.predict(test_image)
result
if result[0][0]>result[0][1]:
  print("animal")
else:
  print("human")
#training_set.class_indicesyy
  
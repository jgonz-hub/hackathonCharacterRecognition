from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.models import load_model
from keras import backend as K
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import os
from imutils import paths
import glob
import numpy as np
import cv2
import imutils
import argparse

file_names = []
directory_path = 'hackathonCharacterRecognition/all_images'
all_files = os.listdir(directory_path)
suffixes = ('.jpg', '.jpeg')
for file in all_files:
    if file.endswith(suffixes):
        file_names.append(file)

sorted_file_names = sorted(file_names)

resized_images = []

desired_width = 256
desired_height = 256

#prints unresized images (=images of different sizes) sorted by filename
# for i in range(0, len(sorted_file_names)):
#     image = cv2.imread("/Users/astroworld97/Desktop/hackathon_fresher/hackathonCharacterRecognition/all_images/" + file_names[i])
#     cv2.imshow('',image)
#     cv2.waitKey(0)

base_directory = os.getcwd() # Assuming the project's base directory is the current working directory
directory_path = os.path.join(base_directory, 'hackathonCharacterRecognition/')

for file_path in sorted_file_names:
    file_path = directory_path + "all_images/" + file_path
    image = cv2.imread(file_path)
    resized_image = cv2.resize(image, (desired_width, desired_height))
    resized_images.append(resized_image)

#prints resized images (=images of all the same size) sorted by filename
# for image in resized_images:
#     cv2.imshow('',image)
#     cv2.waitKey(0)

csv_filename = directory_path + "assigned_classes.csv"
df = pd.read_csv(csv_filename)
sorted_df = df.sort_values(by="img")

classes_list = []
for index, row in sorted_df.iterrows():
    if row["Spongebob"] == 1:
        classes_list.append(0)
    else:
        classes_list.append(1)

classes_df = pd.DataFrame(classes_list)

arr = np.array(resized_images)
data_filepath = directory_path + "Data.npy"
np.save(data_filepath, arr)

Data = np.load(data_filepath)

X_train, X_test, y_train, y_test = train_test_split(Data/255.,classes_df,test_size=0.1,random_state=0)

y_train_cnn = utils.to_categorical(y_train)
y_test_cnn = utils.to_categorical(y_test)
num_classes = y_test_cnn.shape[1]

X_train_cnn = X_train.reshape(X_train.shape[0], 256,256,3).astype('float32')
X_test_cnn = X_test.reshape(X_test.shape[0], 256, 256,3).astype('float32')

def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(256,256,3), activation = 'relu'))
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # the CONV CONV POOL structure is popularized in during ImageNet 2014
    model.add(Dropout(0.25)) # this thing called dropout is used to prevent overfitting
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))


    model.add(Flatten())


    model.add(Dropout(0.5))

    model.add(Dense(2, activation= 'softmax'))

    optimizer = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
    return model

print("[INFO] creating model...")
model = cnn_model()
# Fit the model
print("[INFO] training model...")
# X_train_cnn = np.transpose(X_train_cnn, (0, 2, 3, 1))
# X_train_cnn = np.transpose(X_train_cnn, (0, 2, 3, 1))
# y_train_cnn = np.transpose(y_train_cnn, (0, 2, 3, 1))
records = model.fit(X_train_cnn, y_train_cnn, validation_split=0.1, epochs=5, batch_size=16)
# Final evaluation of the model
print("[INFO] evaluating model...")
scores = model.evaluate(X_test_cnn, y_test_cnn, verbose=0)
print('Final CNN accuracy: ', scores[1])

print("[INFO] saving model...")
model.save(directory_path + 'my_model.keras')

cnn_probab = model.predict(X_test_cnn, batch_size=32, verbose=0)

# extract the probability for the label that was predicted:
p_max = np.amax(cnn_probab, axis=1)

plt.hist(p_max, density=True, bins=list(np.linspace(0, 1, 11)))
plt.xlabel('p of predicted class');

N = 5
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), records.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), records.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), records.history["accuracy"], label="train_accuracy ")
plt.plot(np.arange(0, N), records.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()

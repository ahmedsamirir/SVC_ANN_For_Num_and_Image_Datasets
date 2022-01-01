

# -*- coding: utf-8 -*-

# importing basic Packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import matplotlib.pyplot as plt
#%matplotlib inline
from tqdm import tqdm
#import sklearn

# Set data directory to DATADIR variable and labels of color set to CATEGORIES variable.
DATADIR = "Datasets/ColorClassification"
CATEGORIES = ["Black", "Blue", "Brown", "Green", "orange", "red", "Violet", "White", "Yellow"]
IMG_SIZE=50

# Ex. of an sample image is shown below
for category in CATEGORIES:
    #os.path.join combines path names into one complete path.
    path = os.path.join(DATADIR, category) # path to colors dirs
    for category in CATEGORIES:
        path=os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img)) #cause img is named by numbers
        plt.imshow(img_array)
        plt.show()
        break
    break

# Create training dataset
training_data=[]
def create_training_data():
    for category in CATEGORIES:
        path=os.path.join(DATADIR, category)
        class_num=CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img))
                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_data()

# print the length of training dataset
print(len(training_data))

# storing trainig length for further use.
lenOfTrainingImages = len(training_data)

# for image to be trained we have to convert the image to a array form so,that our model can train on it...!!
# and X should be of type (training_data_length , -1) because SVM takes 2D input to train
x=[]
y=[]

for categories, label in training_data:
    x.append(categories)
    y.append(label)
#print(x)
x= np.array(x).reshape(lenOfTrainingImages,-1)
#print(x)


# Show the shap on X
x.shape

# flattening the array
x = x/255.0

# Ex. of flattened array...
x[1]


# make y in array form
y=np.array(y)
y.shape

################################################################################

###ANN


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split,learning_curve

X_train, X_test, Y_train,Y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#X_train , X_test = X_train / 200.0 , X_test / 200.0 
x.shape
y.shape
X_train.shape
Y_train.shape
ann = models.Sequential([
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(9, activation='softmax')    
    ])

ann.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

History = ann.fit(X_train, Y_train, epochs = 135, batch_size = 10, verbose = 1,validation_split = 0.1)

print('############################################')
###lose curve
ann.summary()
print(History.history.keys())
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc = 'upper right')
plt.show()

print('############################################')

print(ann.evaluate(X_test , Y_test))

print('############################################')

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X_train, Y_train, clf=ann, zoom_factor=1)
plt.show()


print('############################################')

print(ann.metrics_names)
y_pred = ann.predict(X_test)
##ROC Curve
from sklearn.metrics import roc_curve, auc
y_pred = ann.predict(X_test).ravel()
# instantiating the roc_cruve
fpr,tpr,threshols=roc_curve(Y_test,y_pred)

# plotting the curve
plt.plot([0,1],[0,1],"k--",'r+')
plt.plot(fpr,tpr,label='ANN')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ANN ROC Curve")
plt.show()


from sklearn.metrics import confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title="Confusion Matrix",
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalize can be applied by setting 'normalize = True'
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")

    else:
        print("Confusion matrix without normalization")

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cm)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


p_test = ann.predict(X_test).argmax(axis=1)
cm = confusion_matrix(Y_test, p_test)
plot_confusion_matrix(cm, list(range(10)))
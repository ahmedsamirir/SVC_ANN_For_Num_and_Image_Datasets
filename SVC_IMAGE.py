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
IMG_SIZE=96

# Ex. of an sample image is shown below
for category in CATEGORIES:
    path = os.path.join(DATADIR, category) # path to colors dirs
    for category in CATEGORIES:
        path=os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img))
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
x= np.array(x).reshape(lenOfTrainingImages,-1)
##x = tf.keras.utils.normalize(x, axis = 1)

# Show the shap on X
x.shape

# flattening the array
x = x/255.0

# Ex. of flattened array...
x[1]

# make y in array form
y=np.array(y)
y.shape


##############################################################################
#SVC model
from sklearn.svm import SVC
import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.metrics import roc_curve
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import train_test_split,learning_curve
import matplotlib.pyplot as plt
datadir = "Datasets/ColorClassification"
catagories = ["BLACK","BLUE"]

train_data=[]
def create_train_data():
    for category in catagories:
        path=os.path.join(datadir, category)
        class_num=catagories.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img))
                new_array=cv2.resize(img_array,(96,96))
                train_data.append([new_array,class_num])
            except Exception as e:
                pass
create_train_data()

lenofimage = len(train_data)

X=[]
y=[]

for categories, label in train_data:
    X.append(categories)
    y.append(label)
X= np.array(X).reshape(lenofimage,-1)
X=X/255.0
Y=np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.33 )

print("__")
print("train data shape:",X_train.shape)
print("test data shape:",X_test.shape)

model2 = SVC(C=50,kernel="rbf", random_state=40)
model2.fit(X_train,y_train)
print("__")
print("The accuracy of SVC is :",model2.score(X_test,y_test))
cm2 = confusion_matrix(y_test,model2.predict(X_test))
plt.figure(figsize=(7,5))
sn.heatmap(cm2,annot=True)
plt.xlabel("predict")
plt.ylabel("truth")


from sklearn.metrics import roc_curve, auc ,accuracy_score
ylabel = model2.predict(X_test).ravel()
nn_fpr_keras, nn_tpr_keras, nn_thresholds_keras = roc_curve(y_test  , ylabel)
auc_keras = auc(nn_fpr_keras, nn_tpr_keras)
plt.plot(nn_fpr_keras, nn_tpr_keras, marker='.', label='SVM' % auc_keras)
accuracy_score(y_test, ylabel)


####Learing Curve for training dataset
train_sizes, train_scores, test_scores = learning_curve(model2, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.subplots(1, figsize=(10,10))
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()



####Learing Curve for testing dataset
train_sizes, train_scores, test_scores = learning_curve(model2, X_test, y_test, cv=4, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.subplots(1, figsize=(10,10))
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()






print("The loss curve is :",zero_one_loss(y_test,model2.predict(X_test),normalize=False))
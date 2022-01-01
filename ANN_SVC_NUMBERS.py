#-*- coding: utf-8 -*-

import os #using files
import numpy as np #handle array or matrix
import pandas as pd #easy use with ds and loading
import matplotlib.pyplot as plt #data summary
import seaborn as sns #//
import matplotlib.pyplot as plt
import nltk
from sklearn.preprocessing import LabelEncoder
# Importing data into python from the given csv file
dataset= pd.read_csv('Datasets/fake_job_postings.csv')
#first 5 columns
dataset.head()
#last 5 columns
dataset.tail()
#columns' type
dataset.dtypes
#when it runs if numbers of (non-null) > num of entries ,this means that there is null 
dataset.info() 

#how many values is null
dataset.isnull().sum()
db_1 = dataset.drop(["job_id","title","location","department","salary_range","company_profile","benefits","function"], axis = 'columns')
dataset

db_1.columns
Imbalance = sns.countplot(dataset['fraudulent'])

db_new = db_1.dropna()
db_1.shape
db_new.shape

db_new.isnull().sum()

Imbalance = sns.countplot(db_new['fraudulent'])


db_new.info()
le = LabelEncoder()
'''############### Converting industry object column to numeric type##############'''
le_description =LabelEncoder()
db_new['description_num'] = le_description.fit_transform(db_new['description'])
db_new = db_new.drop(["description"], axis = 'columns')
db_new.info()
'''############### Converting requirements object column to numeric type##############'''
le_requirements =LabelEncoder()
db_new['requirements_num'] = le_requirements.fit_transform(db_new['requirements'])
db_new = db_new.drop(["requirements"], axis = 'columns')
db_new.info()
'''############### Converting employment_type object column to numeric type##############'''
le_employment_type = LabelEncoder()
db_new['employment_type_num'] = le_employment_type.fit_transform(db_new['employment_type'])
db_new.info()
db_new = db_new.drop(["employment_type"], axis = 'columns')
'''############### Converting required_experience object column to numeric type##############'''
le_required_experience =LabelEncoder()
db_new['required_experience_num'] = le_required_experience.fit_transform(db_new['required_experience'])
db_new.info()
db_new = db_new.drop(["required_experience"], axis = 'columns')

'''############### Converting required_education object column to numeric type##############'''
Le_required_education =LabelEncoder()
db_new['required_education_num'] = Le_required_education.fit_transform(db_new['required_education'])
db_new = db_new.drop(["required_education"], axis = 'columns')
db_new.info()
'''############### Converting industry object column to numeric type##############'''
le_industry =LabelEncoder()
db_new['industry_num'] = le_industry.fit_transform(db_new['industry'])
db_new = db_new.drop(["industry"], axis = 'columns')
db_new.info()
''' Finding correlation '''
correlation = db_new.corr()

db_new = db_new.drop(["employment_type_num"], axis ="columns")
correlation = db_new.corr()

db_new.info()

'''#################################################################'''
'''#################################################################'''
'''###### SVC MODEL #######'''
from sklearn.svm import SVC
import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.metrics import roc_curve
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import train_test_split,learning_curve
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

###define inputs and outputs sets
X = db_new.drop('fraudulent', axis = 1)
y = db_new['fraudulent']

#####split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train.shape
X_test.shape
y_train.shape
y_test.shape

##feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



###model training
from sklearn.svm import SVC
#The C parameter tells the SVM optimization how much you want to avoid misclassifying each training example.
#'rbf' -> the radial basis function kernel / it is commonly used in support vector machine classification.
clf = SVC(kernel='rbf' , C =70 , random_state=40) 
fitting=clf.fit(X_train, y_train)
clf.fit(X_train, y_train)


clf.score(X_train, y_train) 

y_pred = clf.predict(X_test)
print("The test accuracy score of SVM is ", accuracy_score(y_test, y_pred))


#####confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


##ROC Curve
from sklearn.metrics import roc_curve
y_pred = clf.predict(X_test).ravel()



####Learing Curve for training dataset
train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
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
train_sizes, train_scores, test_scores = learning_curve(clf, X_test, y_test, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
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

'''###########################################---ANN---#######################################'''

from keras.layers import Dense , Dropout
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD

mnist=db_new
X = db_new.drop('fraudulent', axis = 1)
y = db_new['fraudulent']

#####split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


X_train , X_test = X_train / 255.0 , X_test / 255.0 

#Dropout is a technique used to prevent a model from overfitting.
#Dense is used to create fully connected layers, in which every output depends on every input.
#ReLU stands for Rectified Linear Unit. The main advantage of using the ReLU function over other activation functions is that it does not activate all the neurons at the same time.
#Softmax is an activation function that scales numbers/logits into probabilities. The output of a Softmax is a vector (say v ) with probabilities of each possible outcome
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(1680 , 1050)) ,tf.keras.layers.Dense(128 , activation='relu') ,tf.keras.layers.Dropout(0.2) ,tf.keras.layers.Dense(10 , activation='softmax')])
####
#calculate how often predictions equal to labels

# Set random seed
tf.random.set_seed(42)

# 1. Create the model using the Sequential API
#The “Sequential API” is one of the 3 ways to create a Keras model with TensorFlow 2.0. A sequential model, as the name suggests, allows you to create models layer-by-layer in a step-by-step fashion.
model= tf.keras.Sequential([tf.keras.layers.Dense(1)])

# 2. Compile the model
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.SGD(),metrics=['accuracy'])

# 3. Fit the model
model.fit(X, y, epochs=5)

# Train our model for longer (more chances to look at the data)
model.evaluate(X, y)


#####confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


##ROC Curve
from sklearn.metrics import roc_curve, auc
y_pred = clf.predict(X_test).ravel()
nn_fpr_keras, nn_tpr_keras, nn_thresholds_keras = roc_curve(y_test  , y_pred)
auc_keras = auc(nn_fpr_keras, nn_tpr_keras)
plt.plot(nn_fpr_keras, nn_tpr_keras, marker='.', label='ANN' % auc_keras)



####Learing Curve for training dataset
train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
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
train_sizes, train_scores, test_scores = learning_curve(clf, X_test, y_test, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
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














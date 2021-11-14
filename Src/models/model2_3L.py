from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.callbacks import ModelCheckpoint
import seaborn as sns
from keras.optimizers import Adam
import pickle
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer
import os

# fix random seed for reproducibility
np.random.seed(7)

# load dataset
dataset = np.genfromtxt("covid_filtered_1-5_allMin3.csv", delimiter=",", encoding="utf8")
dataset = dataset[1:, :]
np.random.shuffle(dataset)

# split into input and output variables
df_label = dataset[:, 23]
label = []
for lab in df_label:
    if lab == 1:
        label.append([0])  # class 1
    elif lab == 2 or lab == 3:
        label.append([1])  # class 23
    elif lab == 4 or lab == 5:
        label.append([2])  # class 45
    else:
        print("DATA ERROR")
inputColumns = [0, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
label = np.array(label)
xFit, xTest, yFit, yTest = train_test_split(dataset[:, inputColumns], label, test_size=0.3, random_state=42,
                                            stratify=label)

'''
# test:
xTest_c1 = []
yTest_c1 = []
xTest_c23 = []
yTest_c23 = []
xTest_c45 = []
yTest_c45 = []
for i in range(len(yTest)):
    if yTest[i][0] == 1:  # class 1
        xTest_c1.append(xTest[i])
        yTest_c1.append(yTest[i])
    elif yTest[i][1] == 1:  # class 2-3
        xTest_c23.append(xTest[i])
        yTest_c23.append(yTest[i])
    elif yTest[i][2] == 1:  # class 4-5
        xTest_c45.append(xTest[i])
        yTest_c45.append(yTest[i])
xTest_c1 = numpy.array(xTest_c1)
yTest_c1 = numpy.array(yTest_c1)
xTest_c23 = numpy.array(xTest_c23)
yTest_c23 = numpy.array(yTest_c23)
xTest_c45 = numpy.array(xTest_c45)
yTest_c45 = numpy.array(yTest_c45)
'''

parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 50,
              'min_samples_split': 10,
              'max_features': 'sqrt',
              'max_depth': 6,
              'max_leaf_nodes': None}

RF_model = RandomForestClassifier(**parameters)
yFit = np.array(yFit).ravel()
RF_model.fit(xFit, yFit)
RF_predictions = RF_model.predict(xTest)
score = accuracy_score(yTest, RF_predictions)
print(score)

from sklearn import tree
import matplotlib.pyplot as plt

fn = ['sex', 'HSD', 'entry_month', 'symptoms_month', 'pneumonia', 'age_group', 'pregnancy', 'diabetes',
      'copd', 'asthma', 'immsupr', 'hypertension', 'other_disease', 'cardiovascular', 'obesity',
      'renal_chronic', 'tobacco', 'contact_other_covid']
cn = ['Low', 'Middle', 'High']
fig = plt.figure(figsize=(35, 6), dpi=900)
tree.plot_tree(RF_model.estimators_[0],
               feature_names=fn,
               class_names=cn,
               filled=True,
               rounded=True,
               precision=2,
               fontsize=4)
fig.savefig('rf_individualtree.png')
'''
# Get and reshape confusion matrix data
matrix = confusion_matrix(yTest, RF_predictions)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16, 7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size': 10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['Low severity', 'Medium severity', 'High severity']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()


# create model
model = Sequential()
model.add(Dense(729, input_dim=len(inputColumns), activation='sigmoid'))
model.add(Dense(243, activation='sigmoid'))
model.add(Dense(81, activation='sigmoid'))
model.add(Dense(27, activation='sigmoid'))
model.add(Dense(9, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.002), metrics=['accuracy'])

# Fit the model (train the model)
model.fit(xFit, yFit, epochs=1000, batch_size=50)

# evaluate the model
print("\n-------------------------------------------------------")
print("\ntotal(%i):" % len(xTest))
scores = model.evaluate(xTest, yTest)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# test:
print("\nclass1(%i):" % len(xTest_c1))
scores = model.evaluate(xTest_c1, yTest_c1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

print("\nclass23(%i):" % len(xTest_c23))
scores = model.evaluate(xTest_c23, yTest_c23)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

print("\nclass45(%i):" % len(xTest_c45))
scores = model.evaluate(xTest_c45, yTest_c45)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
'''

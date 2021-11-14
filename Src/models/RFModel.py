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

# fix random seed for reproducibility
np.random.seed(0)

# Read dataset
db = pd.read_csv("covid_filtered_1-5_allMin3.csv")

# Take features columns index
inputColumns = [0, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
class1_db = db.loc[db.ill_level == 1]
class2_db = db.loc[db.ill_level == 2]
class3_db = db.loc[db.ill_level == 3]
class4_db = db.loc[db.ill_level == 4]
class5_db = db.loc[db.ill_level == 5]

percentage = 0.7

# Shuffle rows
db1_train = class1_db.sample(frac=percentage, random_state=99)
db1_test = class1_db.loc[~class1_db.index.isin(db1_train.index), :]
db2_train = class2_db.sample(frac=percentage, random_state=99)
db2_test = class2_db.loc[~class2_db.index.isin(db2_train.index), :]
db3_train = class3_db.sample(frac=percentage, random_state=99)
db3_test = class3_db.loc[~class3_db.index.isin(db3_train.index), :]
db4_train = class4_db.sample(frac=percentage, random_state=99)
db4_test = class4_db.loc[~class4_db.index.isin(db4_train.index), :]
db5_train = class5_db.sample(frac=percentage, random_state=99)
db5_test = class5_db.loc[~class5_db.index.isin(db5_train.index), :]

li_train = [db1_train, db2_train, db3_train, db4_train, db5_train]
li_test = [db1_test, db2_test, db3_test, db4_test, db5_test]

db_train = pd.concat(li_train)
db_test = pd.concat(li_test)

db_train = db_train.sample(frac=1, random_state=99)
db_test = db_test.sample(frac=1, random_state=99)

xTrain = db_train.iloc[:, inputColumns]
db_train_label = db_train.iloc[:, 23]

xTest = db_test.iloc[:, inputColumns]
db_test_label = db_test.iloc[:, 23]

yTrain, yTest = [], []
for lab in db_train_label:
    if lab in [1, 2]:
        yTrain.append([0])  # class 1
    elif lab in [3, 4, 5]:
        yTrain.append([1])  # class 2
    else:
        print("DATA ERROR")

for lab in db_test_label:
    if lab in [1, 2]:
        yTest.append([0])  # class 1
    elif lab in [3, 4, 5]:
        yTest.append([1])  # class 2
    else:
        print("DATA ERROR")

yTrain = np.array(yTrain).ravel()
yTest = np.array(yTest).ravel()

parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 50,
              'min_samples_split': 10,
              'max_features': 'sqrt',
              'max_depth': 6,
              'max_leaf_nodes': None}

RF_model = RandomForestClassifier(**parameters)

RF_model.fit(xTrain, yTrain)
RF_predictions = RF_model.predict(xTest)
score = accuracy_score(yTest ,RF_predictions)
print(score)
print(len(yTest))
print(confusion_matrix(yTest, RF_predictions))

from sklearn import tree
import matplotlib.pyplot as plt

fn = ['sex', 'HSD', 'entry_month', 'symptoms_month', 'pneumonia', 'age_group', 'pregnancy', 'diabetes',
      'copd', 'asthma', 'immsupr', 'hypertension', 'other_disease', 'cardiovascular', 'obesity',
      'renal_chronic', 'tobacco', 'contact_other_covid']
cn = ['Low', 'High']
fig = plt.figure(figsize=(35, 6), dpi=900)
tree.plot_tree(RF_model.estimators_[0],
               feature_names=fn,
               class_names=cn,
               filled=True,
               rounded=True,
               precision=2,
               fontsize=4)
fig.savefig('rf_individualtree2L.png')
'''
# Get and reshape confusion matrix data
matrix = confusion_matrix(yTest, RF_predictions)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['Low severity', 'High severity']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()

model = Sequential()
model.add(Dense(24, input_dim=len(inputColumns), activation='sigmoid'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.002), metrics=['binary_accuracy'])

epoches_num = 20
batch = 20

# Fit the model (train the model)
history = model.fit(X, y, epochs=epoches_num, batch_size=batch, shuffle=True)

# Plot graphs
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1, epoches_num + 1)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc_train = history.history['accuracy']
acc_val = history.history['val_accuracy']
epochs = range(1, epoches_num + 1)
plt.plot(epochs, acc_train, 'g', label='Training accuracy')
plt.plot(epochs, acc_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
'''
# Uncomment if you want to export model
"""model.save('model')
with open('config.X', 'wb') as config_X_file:
    pickle.dump(X, config_X_file)"""

# Uncomment for testing explainability
"""features_names = ['sex', 'HSD', 'entry_month', 'symptoms_month', 'pneumonia', 'age_group', 'pregnancy', 'diabetes',
                      'copd', 'asthma', 'immsupr', 'hypertension', 'other_disease', 'cardiovascular', 'obesity',
                      'renal_chronic', 'tobacco', 'contact_other_covid']

ls = []
explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X), feature_names=features_names,
                                                   verbose=True, class_names=['Sick'], mode='classification',
                                                   categorical_features=features_names)
ts = [1, 10, 1, 1, 1, 6, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
ls.append(ts)
ls = np.array(ls)
prediction = model.predict_classes(ls)
print('Model Class: %s' % prediction[0])
exp = explainer.explain_instance(ls[0], model.predict, num_features=len(features_names), labels=[0])
exp.as_pyplot_figure(label=0)
#exp.as_pyplot_figure(label=1)
exp.show_in_notebook(show_table=True, show_all=False)"""

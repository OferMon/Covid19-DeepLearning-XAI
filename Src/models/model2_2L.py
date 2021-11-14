import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import pickle
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer


# fix random seed for reproducibility
np.random.seed(0)

# Read dataset
db = pd.read_csv("covid_filtered_1-5_allMin5.csv")

# Take features columns index
inputColumns = [0, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
class1_db = db.loc[db.ill_level == 1]
class2_db = db.loc[db.ill_level == 2]
class3_db = db.loc[db.ill_level == 3]
class4_db = db.loc[db.ill_level == 4]
class5_db = db.loc[db.ill_level == 5]

# Shuffle rows
db1_train = class1_db.sample(frac=1, random_state=99)
db2_train = class2_db.sample(frac=1, random_state=99)
db3_train = class3_db.sample(frac=1, random_state=99)
db4_train = class4_db.sample(frac=1, random_state=99)
db5_train = class5_db.sample(frac=1, random_state=99)

li_train = [db1_train, db2_train, db3_train, db4_train, db5_train]

db_train = pd.concat(li_train)
db_train = db_train.sample(frac=1, random_state=99)

X = db_train.iloc[:, inputColumns]

# Take output column
db_train_label = db_train.iloc[:, 23]

y = []
for lab in db_train_label:
    if lab in [1, 2]:
        y.append([0])  # class 1
    elif lab in [3, 4, 5]:
        y.append([1])  # class 2
    else:
        print("DATA ERROR")

y = np.array(y)

model = Sequential()
model.add(Dense(24, input_dim=len(inputColumns), activation='sigmoid'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.002), metrics=['binary_accuracy'])

epoches_num = 200
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

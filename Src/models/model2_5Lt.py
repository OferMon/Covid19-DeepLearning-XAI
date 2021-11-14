import numpy
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import pickle

# fix random seed for reproducibility
numpy.random.seed(7)

# load dataset
dataset = numpy.genfromtxt("covid_filtered_1-5_allMin5.csv", delimiter=",", encoding="utf8")
dataset = dataset[1:, :]
numpy.random.shuffle(dataset)

# split into input and output variables
df_label = dataset[:, 23]
label = []
for lab in df_label:
    if lab == 1:
        label.append([1, 0, 0, 0, 0])  # class 1
    elif lab == 2:
        label.append([0, 1, 0, 0, 0])  # class 2
    elif lab == 3:
        label.append([0, 0, 1, 0, 0])  # class 3
    elif lab == 4:
        label.append([0, 0, 0, 1, 0])  # class 4
    elif lab == 5:
        label.append([0, 0, 0, 0, 1])  # class 5
    else:
        print("DATA ERROR")
inputColumns = [0, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

label = numpy.array(label)
# xFit, xTest, yFit, yTest = train_test_split(dataset[:, inputColumns], label, test_size=0.33, random_state=42)
xFit = dataset[:, inputColumns]
yFit = label

# test:
xTest = dataset[:, inputColumns]
yTest = label
xTest_c1 = []
yTest_c1 = []
xTest_c2 = []
yTest_c2 = []
xTest_c3 = []
yTest_c3 = []
xTest_c4 = []
yTest_c4 = []
xTest_c5 = []
yTest_c5 = []
for i in range(len(yTest)):
    if yTest[i][0] == 1:  # class 1
        xTest_c1.append(xTest[i])
        yTest_c1.append(yTest[i])
    elif yTest[i][1] == 1:  # class 2
        xTest_c2.append(xTest[i])
        yTest_c2.append(yTest[i])
    elif yTest[i][2] == 1:  # class 3
        xTest_c3.append(xTest[i])
        yTest_c3.append(yTest[i])
    elif yTest[i][3] == 1:  # class 4
        xTest_c4.append(xTest[i])
        yTest_c4.append(yTest[i])
    elif yTest[i][4] == 1:  # class 5
        xTest_c5.append(xTest[i])
        yTest_c5.append(yTest[i])
xTest_c1 = numpy.array(xTest_c1)
yTest_c1 = numpy.array(yTest_c1)
xTest_c2 = numpy.array(xTest_c2)
yTest_c2 = numpy.array(yTest_c2)
xTest_c3 = numpy.array(xTest_c3)
yTest_c3 = numpy.array(yTest_c3)
xTest_c4 = numpy.array(xTest_c4)
yTest_c4 = numpy.array(yTest_c4)
xTest_c5 = numpy.array(xTest_c5)
yTest_c5 = numpy.array(yTest_c5)

# create model
model = Sequential()
model.add(Dense(500, input_dim=len(inputColumns), activation='sigmoid'))
model.add(Dense(250, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(25, activation='sigmoid'))
model.add(Dense(5, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.002), metrics=['accuracy'])

# Fit the model (train the model)
model.fit(xFit, yFit, epochs=1000, batch_size=50)

model.save('model5')
with open('config5.X', 'wb') as config_X_file:
    pickle.dump(xFit, config_X_file)

# evaluate the model
print("\n-------------------------------------------------------")
print("\ntotal(%i):" % len(xTest))
scores = model.evaluate(xTest, yTest)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# test:
print("\nclass1(%i):" % len(xTest_c1))
scores = model.evaluate(xTest_c1, yTest_c1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

print("\nclass2(%i):" % len(xTest_c2))
scores = model.evaluate(xTest_c2, yTest_c2)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

print("\nclass3(%i):" % len(xTest_c3))
scores = model.evaluate(xTest_c3, yTest_c3)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

print("\nclass4(%i):" % len(xTest_c4))
scores = model.evaluate(xTest_c4, yTest_c4)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

print("\nclass5(%i):" % len(xTest_c5))
scores = model.evaluate(xTest_c5, yTest_c5)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
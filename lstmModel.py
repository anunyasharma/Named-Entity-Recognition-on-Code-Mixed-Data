import preprocess
import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import f1_score
preprocess.createNumFeatures()
dataset = read_csv('featureVector.csv', header=0)
val = dataset.values
val = val.astype('float32')
val = np.nan_to_num(val)

X = val[:,:32]
Y = val[:,32]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
print(X.shape)

model = Sequential()
model.add(LSTM(100, input_shape=(32, 1)))
model.add(Dropout(0.3))
model.add(Dense(7,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
model.fit(X, Y, epochs=5, batch_size=32, validation_split = 0.2, verbose=1)
model.summary()

#no validation data
Y_pred = model.predict(X)
Y_pred_classes = np.argmax(Y_pred, axis=1)
f1 = f1_score(Y, Y_pred_classes, average='weighted')
print("LSTM model F1 Score: {:.2f}".format(f1))
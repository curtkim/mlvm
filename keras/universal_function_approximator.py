# http://gonzalopla.com/deep-learning-nonlinear-regression/
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

#X = np.array([[0],[0.25],[0.5],[0.75]])
#y = np.array([[0],[1.0/16],[8.0/16],[9.0/16]])
X = np.arange(0,1,0.05).reshape((20,1))
#y = X*X +0.3 * np.sin(X)
y = 0.2+0.4*X*X + 0.3*X*np.sin(15*X)+ 0.05 * np.cos(50*X)


model = Sequential()
'''
model.add(Dense(20, input_dim=1))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
sgd = SGD(lr=0.0001)
model.compile(loss='mean_squared_error', optimizer='adam')

'''
model.add(Dense(128, input_dim=1, activation='relu'))
model.add(Dropout(.2))
model.add(Activation("linear"))
model.add(Dense(64, activation='relu'))
model.add(Activation("linear"))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])

np.random.seed(3)
model.fit(X, y, batch_size=20, nb_epoch=1000)
print(model.predict_proba(X))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.plot(X, model.predict_proba(X))
plt.plot(X, y, color='red')
plt.savefig("universal_function_approximator.png")

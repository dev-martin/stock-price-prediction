import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import keras.backend  as K

def rmse(y_pred,y_true):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

def create_dataset(df,timestep):
    x = []
    y = []
    for i in range(timestep, df.shape[0]):
        x.append(df[i-timestep:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y

# Load in dataset
df = pd.read_csv('SP500.csv')
# print(df.head())

df = df['Value'].values
df = df.reshape(-1, 1)
# print(df.shape)
# print(df[:5])

dataset_train = np.array(df[:int(df.shape[0]*0.8)])
dataset_test = np.array(df[int(df.shape[0]*0.8)-50:])
# print(dataset_train.shape)
# print(dataset_test.shape)
# print(dataset_train[:5])
# print(dataset_test[:5])



# Preprocessing data and feature extraction
scaler = MinMaxScaler(feature_range=(0,1))
dataset_train = scaler.fit_transform(dataset_train)
# print(dataset_train[:5])
dataset_test = scaler.transform(dataset_test)
# print(dataset_test[:5])


timestep = 5
x_train, y_train = create_dataset(dataset_train,timestep)
# print(x_train[:1])
# print(y_train[:1])

x_test, y_test = create_dataset(dataset_test,timestep)
# print(x_test[:2])
# print(y_test[:1])

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Build Model
epochs = 150
neurons = 100
model = Sequential()
model.add(LSTM(units=neurons, input_shape=(x_train.shape[1], 1)))
model.add(Dense(units=1))

model.compile(loss=rmse, optimizer="rmsprop" )
history = model.fit(x_train, y_train, epochs=epochs, batch_size=32,validation_data=(x_test , y_test))
loss = model.evaluate(x_test,y_test)
print(loss)


# Plot results
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))


plt.figure(1)
plt.plot(history.history['loss'],label='train_loss',color='blue')
plt.ylabel(' Loss')
plt.xlabel(' Epoch')
plt.title("Training Loss (Time-Step: "+str(timestep)+" Neurons: "+ str(neurons)+")")
plt.legend()

plt.figure(2)
fig, ax = plt.subplots(figsize=(8,4))
plt.title("Stock Market Prediction  Time-Step: "+str(timestep) +" Neurons: "+ str(neurons) )
ax.plot(y_test_scaled, color='red', label='Real Price')
plt.plot(predictions, color='blue', label='Predicted Price')
plt.legend()

plt.show()

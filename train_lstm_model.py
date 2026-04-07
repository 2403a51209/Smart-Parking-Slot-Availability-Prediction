import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

df = pd.read_csv("parking_dataset.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"]).astype(int) // 10**9

X = df.drop("occupancy", axis=1).values
y = df["occupancy"].values

X = X.reshape((X.shape[0], 1, X.shape[1]))

model = Sequential()
model.add(LSTM(50, input_shape=(1, X.shape[2])))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X, y, epochs=3, batch_size=32)

print("LSTM model trained")

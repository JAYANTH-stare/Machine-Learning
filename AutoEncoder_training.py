import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df1 = pd.read_csv("dataset_final.csv")
df2 = pd.read_csv("second_dataset.csv")
df = pd.concat([df1, df2], ignore_index=True)

df["Time"] = pd.to_datetime(df["Time"], unit="s")

scaler = MinMaxScaler()
df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

X = df.iloc[:, 1:].values

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Print dataset shapes
print(f"Training Data Shape: {X_train.shape}")
print(f"Testing Data Shape: {X_test.shape}")

input_dim = X_train.shape[1]
encoding_dim = 4

input_layer = keras.Input(shape=(input_dim,))
encoded = layers.Dense(encoding_dim, activation="relu")(input_layer)

decoded = layers.Dense(input_dim, activation="sigmoid")(encoded)

autoencoder = keras.Model(input_layer, decoded)

autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.summary()

history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_data=(X_test, X_test)
)

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

autoencoder.save_weights("autoencoder.weights.h5")
print("Model weights saved successfully!")

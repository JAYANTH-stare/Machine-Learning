import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Load both datasets
df1 = pd.read_csv("dataset_final.csv")  # First dataset
df2 = pd.read_csv("second_dataset.csv")  # Second dataset

# Combine datasets
df = pd.concat([df1, df2], ignore_index=True)

# Convert Unix timestamp to datetime format
df["Time"] = pd.to_datetime(df["Time"], unit="s")

# Normalize sensor values (excluding time column)
scaler = MinMaxScaler()
df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

# Display first 5 rows
print(df.head())

# Prepare the dataset (exclude the Time column)
X = df.iloc[:, 1:].values  # Convert to NumPy array

# Split into training (80%) and testing (20%) sets
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Print the shape of training and testing sets
print(f"Training Data Shape: {X_train.shape}")
print(f"Testing Data Shape: {X_test.shape}")

# Load pre-trained Hybrid Model if available
try:
    autoencoder = keras.models.load_model("hybrid_model.keras")
    print("Hybrid model loaded successfully!")
except:
    print("No pre-trained model found. Defining and training the model...")

    # Define Autoencoder model
    input_dim = X_train.shape[1]  # Number of sensor features
    encoding_dim = 4  # Compressed representation size

    # Encoder
    input_layer = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation="relu")(input_layer)

    # Decoder
    decoded = layers.Dense(input_dim, activation="sigmoid")(encoded)

    # Autoencoder model
    autoencoder = keras.Model(input_layer, decoded)
autoencoder = keras.Model(input_layer, decoded)

# Compile the model
autoencoder.compile(optimizer="adam", loss="mse")

# Display model summary
autoencoder.summary()

# Load pre-trained Autoencoder weights if available
try:
    autoencoder.load_weights("autoencoder_weights.keras")
    print("Autoencoder weights loaded successfully!")
except:
    print("No pre-trained weights found. Training the model...")
    history = autoencoder.fit(
        X_train, X_train,  # Input and output are the same
        epochs=50,         # Number of training iterations
        batch_size=32,     # Process 32 samples at a time
        shuffle=True,      # Shuffle data to improve learning
        validation_data=(X_test, X_test)  # Validate on test set
    )
    # Save trained weights
    autoencoder.save_weights("autoencoder_weights.keras")
    print("Autoencoder weights saved successfully!")

# Plot Training Loss
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Reconstruct the test data using the trained autoencoder
X_test_pred = autoencoder.predict(X_test)

# Calculate the reconstruction error
reconstruction_error = np.mean(np.abs(X_test - X_test_pred), axis=1)

# Set the anomaly threshold (mean + 3 standard deviations)
threshold = np.mean(reconstruction_error) + 3 * np.std(reconstruction_error)

# Classify anomalies (1 = anomaly, 0 = normal)
anomalies_autoencoder = reconstruction_error > threshold

# Train Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X_train)
anomalies_iforest = iso_forest.predict(X_test)
anomalies_iforest = anomalies_iforest == -1  # Convert -1 to True (Anomaly)

# Train One-Class SVM
oc_svm = OneClassSVM(nu=0.05, kernel="rbf")
oc_svm.fit(X_train)
anomalies_ocsvm = oc_svm.predict(X_test)
anomalies_ocsvm = anomalies_ocsvm == -1  # Convert -1 to True (Anomaly)

# Combine anomaly detections (Majority Voting)
anomalies_combined = (anomalies_autoencoder.astype(int) + anomalies_iforest.astype(int) + anomalies_ocsvm.astype(int)) >= 2

# Print anomaly counts
print(f"Total anomalies detected: {np.sum(anomalies_combined)}")
print(f"Anomaly threshold: {threshold}")

# Plot reconstruction error distribution
plt.figure(figsize=(10, 5))
plt.hist(reconstruction_error, bins=50, alpha=0.7, label="Reconstruction Error")
plt.axvline(threshold, color="red", linestyle="dashed", linewidth=2, label="Anomaly Threshold")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.legend()
plt.title("Reconstruction Error Distribution")
plt.show()

# Add anomaly labels to the test dataset
df_test = pd.DataFrame(X_test, columns=["Temperature", "Humidity", "Air Quality", "Light", "Loudness"])
df_test["Reconstruction Error"] = reconstruction_error
df_test["Anomaly"] = anomalies_combined.astype(int)  # Convert Boolean to 0/1

# Show some detected anomalies
anomalous_data = df_test[df_test["Anomaly"] == 1]
print("Anomalous Data Samples:")
print(anomalous_data.head())

# Save the trained autoencoder model
autoencoder.save("autoencoder_model.keras")
print("Model saved successfully in Keras format!")

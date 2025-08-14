import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Load the features and labels data
X = np.load('features_min.npy')
y = np.load('labels.npy')
ss = np.load('ssdata.npy')

# # Split the data into training, validation and testing sets
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Split the data into training, validation and testing sets
X_train, X_temp, y_train, y_temp, ss_train, ss_temp = train_test_split(X, y, ss, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test, ss_val, ss_test = train_test_split(X_temp, y_temp, ss_temp, test_size=0.5, random_state=42)

# # Feature scaling using StandardScaler
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)
# X_test_scaled = scaler.transform(X_test) 

# Build the MLP model
model = Sequential()
model.add(Dense(1024, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.3))  # Add dropout with a dropout rate of 0.5 (adjust as needed)
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))  # Add dropout with a dropout rate of 0.5 (adjust as needed)
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))  # Add dropout with a dropout rate of 0.5 (adjust as needed)
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))  # Add dropout with a dropout rate of 0.5 (adjust as needed)
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')

# Train the model and track training history
history = model.fit(X_train, y_train, epochs=100, batch_size=256, validation_data=(X_val, y_val))

# Plot training and validation curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions on the test set
predictions = model.predict(X_test)

# For regression tasks (e.g., predicting a continuous value)
# Evaluate using a relevant metric (e.g., Mean Squared Error)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error on Test Set: {mse}')

# plt.plot(y_test, predictions, 'o')
plt.scatter(ss_test[:,0], ss_test[:,1],color='r',alpha=0.1,label='$S_{filtered}$')
plt.scatter(y_test, predictions,alpha=0.1,label='MLP')
plt.xlim([y_test.min(), y_test.max()])
plt.ylim([y_test.min(), y_test.max()])
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],'orange')
plt.xlabel('Ground Truth, $S_{eff}$')
plt.ylabel('Prediction')
plt.legend()
plt.show()


mape = np.mean(np.abs((y_test.squeeze() - predictions.squeeze()) / np.abs(y_test.squeeze()))) * 100
smape = np.mean(np.abs(2 * (y_test - predictions.squeeze()) / (np.abs(y_test) + np.abs(predictions.squeeze()))) * 100)
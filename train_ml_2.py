import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score
from keras.models import Model
from keras.layers import Input, Conv3D, Flatten, Dense, concatenate, MaxPooling3D
from keras.utils import plot_model

# Load the features and labels data
X_mlp = np.load('features_mlp.npy')
X_cnn = np.load('features_cnn.npy')
y = np.load('labels.npy')
ss = np.load('ssdata.npy')

# Split the data into training, validation and testing sets
X_mlp_train, X_mlp_temp, X_cnn_train, X_cnn_temp, y_train, y_temp, ss_train, ss_temp = train_test_split(X_mlp, X_cnn, y, ss, test_size=0.3, random_state=42)
X_mlp_val, X_mlp_test, X_cnn_val, X_cnn_test, y_val, y_test, ss_val, ss_test = train_test_split(X_mlp_temp, X_cnn_temp, y_temp, ss_temp, test_size=0.5, random_state=42)


# Input for the volume data
volume_input = Input(shape=(3, 3, 3, 6), name='volume_input')
# CNN layers for the volume data
conv_layer_1 = Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same',activation='relu',name='conv_layer_1')(volume_input)
pooling_layer_1 = MaxPooling3D(pool_size=(2, 2, 2),name='pooling_layer')(conv_layer_1)
conv_layer_2 = Conv3D(filters=64, kernel_size=(2, 2, 2), padding='same', activation='relu',name='conv_layer_2')(pooling_layer_1)

cnn_output = Flatten(name='cnn_output')(conv_layer_2)

# Input for the feature vector
feature_input = Input(shape=(9,), name='feature_input')
# MLP layers for the feature vector
mlp_layer_1 = Dense(128, activation='relu', name='mlp_layer_1')(feature_input)
BN_layer_1 = BatchNormalization()(mlp_layer_1)
DP_layer_1 = Dropout(0.5)(BN_layer_1)
mlp_layer_2 = Dense(64, activation='relu',name='mlp_layer_2')(DP_layer_1)
BN_layer_2 = BatchNormalization()(mlp_layer_2)
DP_layer_2 = Dropout(0.5)(BN_layer_2)
mlp_output = Dense(32, activation='relu',name='mlp_output')(DP_layer_2)

# Concatenate the output of CNN and MLP
concatenated_output = concatenate([cnn_output, mlp_output])

# Fully connected layers for regression
# final_layer_1 = Dense(128, activation='relu',name='final_layer_1')(concatenated_output)
# BN_layer_1 = BatchNormalization()(final_layer_1)
# DP_layer_1 = Dropout(0.3)(BN_layer_1)
# final_layer_2 = Dense(64, activation='relu',name='final_layer_2')(DP_layer_1)
# BN_layer_2 = BatchNormalization()(final_layer_2)
# DP_layer_2 = Dropout(0.3)(BN_layer_2)
# final_output = Dense(1, activation='linear', name='output')(DP_layer_2)

final_layer_1 = Dense(128, activation='relu',name='final_layer_1')(concatenated_output)
final_layer_2 = Dense(64, activation='relu',name='final_layer_2')(final_layer_1)
final_output = Dense(1, activation='linear', name='output')(final_layer_2)
# Create the model
model = Model(inputs=[volume_input, feature_input], outputs=final_output)

# Compile the model with an appropriate loss and optimizer for regression
# model.compile(optimizer='adam', loss='mean_squared_error')
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Display the model summary
# model.summary()
plot_model(model, "my_first_model.png")

# Combine scaled volume and feature data
X_train = [X_cnn_train, X_mlp_train]
X_val = [X_cnn_val, X_mlp_val]
X_test = [X_cnn_test, X_mlp_test]

# Train the model
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
r2 = r2_score(y_test,predictions.squeeze())
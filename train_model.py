import tensorflow as tf
from tensorflow import keras
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Load data yang telah diproses
X = pd.read_csv("X_scaled.csv").values
y = pd.read_csv("y.csv").values.ravel()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membangun model ANN
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(len(set(y)), activation='softmax')
])

# Kompilasi model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Melatih model
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=1)

# Simpan model
model.save("model_ann.keras")
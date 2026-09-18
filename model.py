import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(input_shape[1])
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Example training loop
# model = build_model((100, 12))
# model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

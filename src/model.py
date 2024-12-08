import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_model(input_shape: tuple, lstm_units: list = [64, 32], dropout: float = 0.5):
    model = Sequential()
    model.add(LSTM(lstm_units[0], input_shape=input_shape, return_sequences=True))
    model.add(Dropout(dropout))
    for units in lstm_units[1:]:
        model.add(LSTM(units, return_sequences=False))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model

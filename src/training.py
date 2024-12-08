from sklearn.preprocessing import StandardScaler
from src.model import build_model

class SeizureForecaster:
    def __init__(self, prediction_horizon=30, sequence_length=60):
        self.prediction_horizon = prediction_horizon
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()

    def prepare_sequences(self, features, labels):
        X, y = [], []
        for i in range(len(features) - self.sequence_length):
            X.append(features[i:i + self.sequence_length])
            future_window = labels[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]
            y.append(1 if np.any(future_window) else 0)
        return np.array(X), np.array(y)

    def train(self, train_data, val_data, epochs=100, batch_size=32):
        X_train, y_train = train_data
        X_val, y_val = val_data
        self.model = build_model(input_shape=(self.sequence_length, X_train.shape[-1]))
        history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
        return history

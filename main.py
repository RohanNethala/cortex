import pandas as pd
from src.data_processing import process_wearable_data
from src.training import SeizureForecaster
from src.evaluation import evaluate_forecaster

raw_data = pd.read_csv('data/wearable_data.csv')
seizure_labels = pd.read_csv('data/seizure_events.csv')
processed_data = process_wearable_data(raw_data)

forecaster = SeizureForecaster()
X, y = forecaster.prepare_sequences(processed_data, seizure_labels)
train_idx = int(len(X) * 0.8)
X_train, y_train = X[:train_idx], y[:train_idx]
X_val, y_val = X[train_idx:], y[train_idx:]

history = forecaster.train((X_train, y_train), (X_val, y_val))
predictions = forecaster.model.predict(X_val)
random_predictions = np.random.random(len(y_val))

metrics = evaluate_forecaster(y_val, predictions, random_predictions)
print(f"AUC-ROC: {metrics['auc_roc']}")

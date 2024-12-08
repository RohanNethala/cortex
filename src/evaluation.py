from sklearn.metrics import roc_auc_score, precision_recall_curve

def evaluate_forecaster(y_true, y_pred, random_predictions):
    actual_auc = roc_auc_score(y_true, y_pred)
    random_auc = roc_auc_score(y_true, random_predictions)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    return {'auc_roc': actual_auc, 'random_auc': random_auc, 'precision': precision, 'recall': recall, 'thresholds': thresholds}

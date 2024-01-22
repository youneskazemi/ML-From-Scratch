def calculate_accuracy(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)

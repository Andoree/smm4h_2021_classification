from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate_classification(true_y, pred_y):
    precision = precision_score(true_y, pred_y, )
    recall = recall_score(true_y, pred_y, )
    f_measure = f1_score(true_y, pred_y, )
    d = {
        "Precision": precision,
        "Recall": recall,
        "F1": f_measure
    }
    return d


def predict_labels(model, word_seq, classification_threshold):
    predicted_prob = model.predict(word_seq)
    predicted_labels = []
    for pred in predicted_prob:
        label = 1 if pred[0] >= classification_threshold else 0
        predicted_labels.append(label)
    return predicted_labels


import json
from sklearn.metrics import precision_recall_fscore_support

import json
from sklearn.metrics import precision_recall_fscore_support

def predict_best(good_score, medium_score, bad_score):
    scores = [good_score, medium_score, bad_score]
    return int(scores.index(max(scores)) == 0)

def metrics(predictions, actuals):
    accuracy = sum([1 if pred == actual else 0 for pred, actual in zip(predictions, actuals)]) / len(predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(actuals, predictions, labels=[1], average='binary', zero_division=0)
    return accuracy, precision, recall, f1


def calc_accuracy_auto(filepath):
    
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]

    all_predictions = {}
    for item in data:
        scores = item['scores']
        for metric_key, metric_scores in scores.items():
            if metric_key not in all_predictions:
                all_predictions[metric_key] = []
            all_predictions[metric_key].append(
                predict_best(metric_scores['good'], metric_scores['medium'], metric_scores['bad'])
            )

    labels = [1] * len(data)

    print("\nSummary (Accuracy):")
    for metric_name, predictions in all_predictions.items():
        acc, prec, rec, f1 = metrics(predictions, labels)
        count = len(predictions)
        print(f"{metric_name:30}: {acc:.4f} ")



if __name__ == "__main__":

    filepath = "Synthetic_dataset/Synthetic_dataset_KG_Score.jsonl"

    calc_accuracy_auto(filepath)


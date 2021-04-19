import os
from argparse import ArgumentParser

import pandas as pd
from sklearn.metrics import precision_score, f1_score, recall_score, classification_report

METRICS = {"Precision": precision_score, "Recall": recall_score,
           "F-score": f1_score, }


def main():
    parser = ArgumentParser()
    parser.add_argument('--predicted_probs_dir', )
    parser.add_argument('--data_tsv', )
    parser.add_argument('--calculate_metrics', action="store_true",
                        help="Defines whether to calculate P, R, F1")
    parser.add_argument('--probas_fname', default="pred_test_probas.txt")
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output_path', )
    args = parser.parse_args()

    predicted_probs_dir = args.predicted_probs_dir
    decision_threshold = args.threshold
    data_tsv_path = args.data_tsv
    calculate_metrics = args.calculate_metrics
    probas_fname = args.probas_fname
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and not output_dir == '':
        os.makedirs(output_dir)
    predictions = []
    columns = []
    i = 0
    for filename in os.listdir(predicted_probs_dir):
        if filename.startswith("seed"):
            prediction_path = os.path.join(predicted_probs_dir, f"{filename}/{probas_fname}")
            prediction_df = pd.read_csv(prediction_path, sep="\t", encoding="utf-8", header=None, names=["proba"])
            prediction_df[f'p_{i}'] = prediction_df["proba"].apply(lambda x: 1 if x > decision_threshold else 0)
            predicted_labels_df = prediction_df[f'p_{i}']
            predictions.append(predicted_labels_df)
            columns.append(f"p_{i}")
    all_predictions = pd.concat(predictions, axis=1)
    all_predictions['sum'] = (all_predictions >= 0.5).sum(axis=1)
    all_predictions['final_label'] = all_predictions['sum'].apply(lambda x: 1 if x >= len(columns) / 2 else 0)

    data_df = pd.read_csv(data_tsv_path, sep="\t", encoding="utf-8",)
    if calculate_metrics:
        true_labels_df = data_df["class"]
        print(classification_report(true_labels_df, all_predictions['final_label']))
        for metric_name, metric in METRICS.items():
            print(f"{metric_name}", metric(true_labels_df, all_predictions['final_label']))
    data_df['Class'] = all_predictions['final_label']
    data_df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")


if __name__ == '__main__':
    main()

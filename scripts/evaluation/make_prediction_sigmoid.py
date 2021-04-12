import os
from argparse import ArgumentParser

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, f1_score, recall_score, classification_report

METRICS = {"Precision": precision_score, "Recall": recall_score,
           "F-score": f1_score, }


def main():
    parser = ArgumentParser()
    parser.add_argument('--predicted_probs_dir', )
    parser.add_argument('--multilabel_probs_dir', default=None)
    parser.add_argument('--test_data_tsv', )
    parser.add_argument('--dev_data_tsv', )
    parser.add_argument('--calculate_metrics', action="store_true",
                        help="Defines whether to calculate P, R, F1")
    parser.add_argument('--dev_probas_fname', default="pred_dev_probas.txt")
    parser.add_argument('--test_probas_fname', default="pred_test_probas.txt")
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--output_path', )
    args = parser.parse_args()

    predicted_probs_dir = args.predicted_probs_dir
    multilabel_probs_dir = args.multilabel_probs_dir
    decision_threshold = args.threshold
    dev_data_tsv_path = args.dev_data_tsv
    test_data_tsv_path = args.test_data_tsv
    calculate_metrics = args.calculate_metrics
    dev_probas_fname = args.dev_probas_fname
    test_probas_fname = args.test_probas_fname
    random_state = args.random_state
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and not output_dir == '':
        os.makedirs(output_dir)
    dev_predictions = []
    columns = []
    i = 0
    for filename in sorted(os.listdir(predicted_probs_dir)):
        if filename.startswith("seed"):
            prediction_path = os.path.join(predicted_probs_dir, f"{filename}/{dev_probas_fname}")
            prediction_df = pd.read_csv(prediction_path, sep="\t", encoding="utf-8", header=None, names=["proba"])
            # prediction_df[f'p_{i}'] = prediction_df["proba"].apply(lambda x: 1 if x > decision_threshold else 0)
            predicted_probas_df = prediction_df[f"proba"]
            dev_predictions.append(predicted_probas_df)
            columns.append(f"p_{i}")
    dev_all_predictions = pd.concat(dev_predictions, axis=1)
    if multilabel_probs_dir is not None:
        dev_multilabel_predictions = []
        ef_columns = []
        i = 0
        multilabel_columns = ["EF", "INF", "ADR", "DI", "Finding"]
        for filename in sorted(os.listdir(multilabel_probs_dir)):
            if filename.startswith("seed"):
                prediction_path = os.path.join(multilabel_probs_dir, f"{filename}/{dev_probas_fname}")
                names = [f"{col}_{i}" for col in multilabel_columns]
                prediction_df = pd.read_csv(prediction_path, sep="\t", encoding="utf-8", header=None, names=names)

                # prediction_df[f'p_{i}'] = prediction_df["proba"].apply(lambda x: 1 if x > decision_threshold else 0)
                predicted_probas_df = prediction_df[f"EF_{i}"]
                dev_multilabel_predictions.append(predicted_probas_df)
                ef_columns.append(f"EF_{i}")
        multilabel_dev_all_predictions = pd.concat(dev_multilabel_predictions, axis=1)
        dev_all_predictions = pd.concat((dev_all_predictions, multilabel_dev_all_predictions), axis=1)

    dev_predictions_numpy = dev_all_predictions.values
    dev_data_df = pd.read_csv(dev_data_tsv_path, sep="\t", encoding="utf-8", quoting=3)
    dev_true_labels = dev_data_df["class"].values
    logistic_regression_model = LogisticRegression(random_state=random_state, max_iter=1000)
    logistic_regression_model.fit(dev_predictions_numpy, dev_true_labels)
    dev_predictions = logistic_regression_model.predict_proba(dev_predictions_numpy)
    best_f1_score = -1.0
    best_decision_threshold = 0.0
    for i in range(1, 100):
        decision_threshold = 0.01 * i
        dev_pred_labels = [1 if x[1] > decision_threshold else 0 for x in dev_predictions]
        dev_f1_score = f1_score(dev_true_labels, dev_pred_labels)
        if dev_f1_score > best_f1_score:
            best_f1_score = dev_f1_score
            best_decision_threshold = decision_threshold
    # best_decision_threshold = 0.5
    dev_predictions = [1 if x[1] > best_decision_threshold else 0 for x in dev_predictions]
    print("Dev:")
    print("Threshold:", best_decision_threshold, "F1", best_f1_score)
    print(classification_report(dev_true_labels, dev_predictions))
    for metric_name, metric in METRICS.items():
        print(f"{metric_name}", metric(dev_true_labels, dev_predictions))
    print('--' * 10)

    # dev_all_predictions['sum'] = (dev_all_predictions >= 0.5).sum(axis=1)
    # dev_all_predictions['final_label'] = dev_all_predictions['sum'].apply(lambda x: 1 if x >= len(columns) / 2 else 0)

    test_data_df = pd.read_csv(test_data_tsv_path, sep="\t", encoding="utf-8", quoting=3)
    test_predictions = []
    columns = []
    i = 0
    for filename in sorted(os.listdir(predicted_probs_dir)):
        if filename.startswith("seed"):
            prediction_path = os.path.join(predicted_probs_dir, f"{filename}/{test_probas_fname}")
            prediction_df = pd.read_csv(prediction_path, sep="\t", encoding="utf-8", header=None, names=["proba"])
            # prediction_df[f'p_{i}'] = prediction_df["proba"].apply(lambda x: 1 if x > decision_threshold else 0)
            predicted_probas_df = prediction_df[f"proba"]
            test_predictions.append(predicted_probas_df)
            columns.append(f"p_{i}")
    test_all_predictions = pd.concat(test_predictions, axis=1)
    if multilabel_probs_dir is not None:
        test_multilabel_predictions = []
        ef_columns = []
        i = 0
        multilabel_columns = ["EF", "INF", "ADR", "DI", "Finding"]
        for filename in sorted(os.listdir(multilabel_probs_dir)):
            if filename.startswith("seed"):
                prediction_path = os.path.join(multilabel_probs_dir, f"{filename}/{test_probas_fname}")
                names = [f"{col}_{i}" for col in multilabel_columns]
                prediction_df = pd.read_csv(prediction_path, sep="\t", encoding="utf-8", header=None, names=names)
                predicted_probas_df = prediction_df[f"EF_{i}"]
                test_multilabel_predictions.append(predicted_probas_df)
                ef_columns.append(f"EF_{i}")
        multilabel_test_all_predictions = pd.concat(test_multilabel_predictions, axis=1)
        test_all_predictions = pd.concat((test_all_predictions, multilabel_test_all_predictions), axis=1)

    test_predictions_numpy = test_all_predictions.values
    test_predictions = logistic_regression_model.predict_proba(test_predictions_numpy)
    test_predictions = [1 if x[1] > best_decision_threshold else 0 for x in test_predictions]
    if calculate_metrics:
        print("Test")
        true_labels_df = test_data_df["class"]
        print(classification_report(true_labels_df, test_predictions))
        for metric_name, metric in METRICS.items():
            print(f"{metric_name}", metric(true_labels_df, test_predictions))

    test_data_df['Class'] = test_predictions
    test_data_df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")


if __name__ == '__main__':
    main()

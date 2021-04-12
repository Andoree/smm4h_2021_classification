import codecs
import configparser
import os
import pandas as pd


def main():
    config = configparser.ConfigParser()
    config.read("train_config.ini")

    results_dir = config["INPUT"]["INPUT_DIR"]
    output_path = config["OUTPUT"]["OUTPUT_PATH"]
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    evaluation_results_list = []
    for name in os.listdir(results_dir):
        if name.startswith('seed'):
            seed = name.split('_')[-1]
            evaluation_file_path = os.path.join(results_dir, name, "evaluation.txt")
            with codecs.open(evaluation_file_path, 'r', encoding="utf-8") as eval_file:
                for i, line in enumerate(eval_file):
                    attrs = line.strip().split(',')
                    model_name = attrs[0]
                    f1_score = float(attrs[-1])
                    recall_score = float(attrs[-2])
                    precision_score = float(attrs[-3])
                    num_epochs = int(attrs[1]) if len(attrs) >= 5 else None
                    dataset = "dev" if i % 2 == 0 else "test"
                    model_dict = {
                        "model": model_name,
                        "dataset": dataset,
                        "num_epochs": num_epochs,
                        "seed": seed,
                        "precision": precision_score,
                        "recall": recall_score,
                        "f1": f1_score
                    }
                    evaluation_results_list.append(model_dict)
    results_df = pd.DataFrame(evaluation_results_list)
    print(results_df)
    avg_quality_df = results_df.groupby(by=["model", "dataset"])["precision", "recall", "f1"].mean()
    avg_quality_df.to_csv(output_path)


if __name__ == '__main__':
    main()

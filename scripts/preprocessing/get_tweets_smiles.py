import os
from argparse import ArgumentParser
from typing import Dict

import numpy as np
import pandas as pd


def smiles_by_drug_ids(drug_ids_str: str, id_to_smiles_mapping: Dict[str, str], drugs_sep: str = '~',
                       smiles_sep: str = '~~~') -> str:
    drugs_list = drug_ids_str.split(drugs_sep)
    smiles_list = []
    for drug_id in drugs_list:
        drug_smiles = id_to_smiles_mapping[drug_id]

        if drug_smiles is not np.nan:
            smiles_list.append(drug_smiles)
    return smiles_sep.join(smiles_list)


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_drugbank', default=r"../../data/drugbank_database.csv")
    parser.add_argument('--input_tweets', default="../../data/smm4h_21_data/en/tweets_w_drugs/train.tsv")
    parser.add_argument('--output_path', default=r"../../data/smm4h_21_data/en/tweets_w_smiles/train.tsv")
    args = parser.parse_args()

    drugbank_path = args.input_drugbank
    tweets_path = args.input_tweets
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and not output_dir == '':
        os.makedirs(output_dir)

    drugbank_df = pd.read_csv(drugbank_path)
    drugbank_id_smiles_df = drugbank_df[["drugbank_id", "smiles"]]
    tweets_df = pd.read_csv(tweets_path, sep='\t', quoting=3)

    drugbank_id_smiles_df.set_index("drugbank_id", inplace=True)

    drugbank_id_smiles_df = drugbank_id_smiles_df.squeeze()
    tweets_df["smiles"] = tweets_df.drug_id.apply(lambda x: smiles_by_drug_ids(x, drugbank_id_smiles_df))
    tweets_df.to_csv(output_path, sep='\t', index=False, quoting=3)


if __name__ == '__main__':
    main()

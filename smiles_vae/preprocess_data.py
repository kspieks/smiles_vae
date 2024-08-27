import argparse
import os
from pprint import pprint

import pandas as pd

from smiles_vae.data.data_classes import PreprocessingArgs
from smiles_vae.data.data_cleaning import clean_smiles
from smiles_vae.data.tokenization import create_vocabulary
from smiles_vae.utils.parsing import read_yaml_file


def clean_data(preprocess_data_args):
    """
    Cleans SMILES in preparation for training a generative model.

    Args:
        preprocess_data_args: dataclass storing arguments for preprocessing the SMILES.
    """

    # read in data as df
    df = pd.read_csv(preprocess_data_args.gen_input_file)

    # clean the SMILES
    df = clean_smiles(df,
                      min_mol_wt=preprocess_data_args.min_mol_wt,
                      max_mol_wt=preprocess_data_args.max_mol_wt,
                      supported_elements=preprocess_data_args.supported_elements,
                      remove_stereo=preprocess_data_args.remove_stereo,
                      )
    
    # save the cleaned SMILES
    directory = os.path.dirname(preprocess_data_args.gen_output_file)
    os.makedirs(directory, exist_ok=True)
    df.to_csv(preprocess_data_args.gen_output_file, index=False)
    # with open(preprocess_data_args.gen_output_file, 'w') as f:
    #     f.write('\n'.join(df.SMILES.values) + '\n')

    # create the vocabulary
    vocab_tokens = create_vocabulary(df.SMILES.values)
    print(f'Vocabulary contains {len(vocab_tokens)} tokens:')
    print('\n'.join(vocab_tokens))

    # save the vocab tokens
    with open(preprocess_data_args.gen_vocab_file, 'w') as f:
        for token in vocab_tokens.keys():
            f.write(token + "\n")


def main():
    parser = argparse.ArgumentParser(description="Script to process data in preparation for training a VAE.")
    parser.add_argument('--yaml_file', required=True,
                        help='Path to yaml file containing arguments for preprocessing.')
    args = parser.parse_args()

    # read yaml file
    print('Using arguments...')
    yaml_dict = read_yaml_file(args.yaml_file)
    pprint(yaml_dict)
    print('\n\n')
    
    # clean and filter the data
    preprocess_data_args = PreprocessingArgs(**yaml_dict['preprocess_data'])
    clean_data(preprocess_data_args)


if __name__ == "__main__":
    main()

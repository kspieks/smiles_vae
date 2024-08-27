from dataclasses import dataclass, field, fields
from typing import Dict, List

MIN_MOL_WT = 200
MAX_MOL_WT = 700
SUPPORTED_ELEMENTS = {
    'H',
    'B', 'C', 'N', 'O', 'F',
    'P', 'S', 'Cl',
    'Br', 'I',
}

@dataclass
class PreprocessingArgs:
    """
    Class to store settings for cleaning SMILES during data pre-processing.

    Args:
        gen_input_file: path to a csv file with SMILES to be cleaned before training a generative model.
        gen_output_file: path to write the cleaned SMILES for training a generative model.
        gen_vocab_file: path to write the vocabulary to.

        min_mol_wt: minimum molecular weight.
        max_mol_wt: maximum molecular weight.
        supported_elements: set of supported atomic symbols.
        remove_stereo: boolean indicating whether to remove stereochemistry.
    """
    gen_input_file: str = 'smiles.csv'
    gen_output_file: str = 'cleaned_mols.csv'
    gen_vocab_file: str = 'vocab.txt'

    min_mol_wt: int = MIN_MOL_WT
    max_mol_wt: int = MAX_MOL_WT
    supported_elements: set = field(default_factory=lambda: SUPPORTED_ELEMENTS)
    remove_stereo: bool = True

    def __post_init__(self):
        for field in fields(self):
            setattr(self, field.name, field.type(getattr(self, field.name)))


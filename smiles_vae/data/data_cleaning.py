"""Functions to clean and filter SMILES during data preprocessing."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from .data_classes import MAX_MOL_WT, MIN_MOL_WT, SUPPORTED_ELEMENTS


def filter_smiles(smi,
                  min_mol_wt=MIN_MOL_WT,
                  max_mol_wt=MAX_MOL_WT,
                  supported_elements=SUPPORTED_ELEMENTS,
                  ):
    """
    Filter SMILES based on molecular weight and atom types.

    Args:
        smi (str): SMILES string.
        min_mol_wt (int): minimum molecular weight.
        max_mol_wt (int): maximum molecular weight.
        supported_elements (set): set of supported elements.

    Returns:
        boolean: whether the SMILES passed the filters.
    """
    mol = Chem.MolFromSmiles(smi)
    
    # reject molecule that is too small or too big
    if not min_mol_wt <= Descriptors.MolWt(mol) <= max_mol_wt:
        return False

    # reject molecule that contains unsupported atom types
    if not all([atom.GetSymbol() in supported_elements for atom in mol.GetAtoms()]):
        return False

    return True

def remove_stereochemistry(smi, canonicalize=True):
    """Remove stereochemistry and optionally canonicalize the SMILES"""
    mol = Chem.MolFromSmiles(smi)
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=canonicalize)


def canonicalize_smiles(smi):
    """Canonicalize the SMILES string."""
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, canonical=True)


def standardize(smi):
    """Clean up SMILES string by neutralizing charges"""
    try:
        mol = Chem.MolFromSmiles(smi)

        # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
        clean_mol = rdMolStandardize.Cleanup(mol)

        # if many fragments, get the "parent" (the actual mol we are interested in)
        # i.e., Returns the largest fragment after doing a cleanup
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)

        # neutralize the molecule
        uncharger = rdMolStandardize.Uncharger()
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
        return Chem.MolToSmiles(uncharged_parent_clean_mol, canonical=True)

    except Exception as e:
        print(f'Error with {smi}')
        print(e)
        return None


def clean_smiles(df,
                 min_mol_wt=MIN_MOL_WT,
                 max_mol_wt=MAX_MOL_WT,
                 supported_elements=SUPPORTED_ELEMENTS,
                 remove_stereo=False,
                 ):
    """
    Cleans the SMILES from a dataframe:
        - remove any SMILES that cannot be rendered by RDKit
        - apply additional filters based on molecular weight and atom type
        - optionally remove stereochemistry
        - SMILES are canonicalized
        - calculate InChI keys and remove duplcate entries
    """
    print('Cleaning dataset...')
    print(f'Dataframe has {len(df)} rows')

    # make sure appropriate columns are present
    if 'SMILES' not in df.columns:
        raise ValueError("Column containing SMILES strings must be labeled as 'SMILES'.")

    # remove any rows that are missing a SMILES string
    num_nan = len(df[df.SMILES.isna()])
    print(f"Removing {num_nan} rows that are missing a SMILES string")
    df = df[~df.SMILES.isna()]

    # remove any SMILES that cannot be rendered by RDKit
    df['invalid_smiles'] = df.SMILES.apply(lambda smi: True if Chem.MolFromSmiles(smi) is None else False)
    df_invalid = df.query('invalid_smiles == True')
    print(f"Removing {len(df_invalid)} invalid SMILES that could not be rendered by RDKit")
    if len(df_invalid):
        for smi in df_invalid.SMILES:
            print(smi)
    df = df.query('invalid_smiles == False').drop('invalid_smiles', axis=1)   # remove the temporary column

    # apply additional filters based on molecular weight and atom type
    kwargs = {
        'min_mol_wt': min_mol_wt,
        'max_mol_wt': max_mol_wt,
        'supported_elements': supported_elements,
    }
    df['filtered_smiles'] = df.SMILES.apply(filter_smiles, **kwargs)
    df_invalid = df.query('filtered_smiles == False')
    print(f"\nRemoving {len(df_invalid)} SMILES that did not pass the filters based on molecular weight and supported atom types")
    if len(df_invalid):
        for smi in df_invalid.SMILES:
            print(smi)
    df = df.query('filtered_smiles == True').drop('filtered_smiles', axis=1)   # remove the temporary column

    # optionally remove stereochemistry
    if remove_stereo:
        print('\nRemoving stereochemistry from all SMILES')
        df.SMILES = df.SMILES.apply(remove_stereochemistry)
    
    # standardize the SMILES
    print('Standardizing the SMILES...')
    df.SMILES = df.SMILES.apply(standardize)
    df = df[~df.SMILES.isna()]

    # get InChI keys and remove duplicate compounds
    print('Calculating InChI keys...')
    df['inchi_key'] = df.SMILES.apply(lambda smi: Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(smi)))
    num_duplicated = df.duplicated(subset=['inchi_key']).sum()
    print(f'Only unique InChI keys will be kept i.e., removing {num_duplicated} compounds that appear multiple times')
    df = df.drop_duplicates(subset='inchi_key')
    print(f'Final cleaned file has {len(df)} rows\n')

    return df

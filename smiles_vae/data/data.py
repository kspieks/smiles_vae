import torch
from torch.utils.data import Dataset

class SMILESDataset(Dataset):
    """
    Custom PyTorch Dataset that takes a file containing SMILES.

    Args:
        file_path: path to a file containing SMILES strings separated by newlines.
        voc: a Vocabulary instance.

    Returns:
        Custom PyTorch dataset.
    """
    def __init__(self, file_path, voc):
        self.voc = voc
        self.smiles = []
        with open(file_path, 'r') as f:
            for line in f:
                self.smiles.append(line.strip())
    
    def __getitem__(self, i):
        smi = self.smiles[i]
        tokenized_smi = self.voc.tokenize(smi, add_bos=False, add_eos=True, add_pad=False)
        encoded_smi = self.voc.encode(tokenized_smi)
        return torch.tensor(encoded_smi, dtype=torch.long)
    
    def __len__(self):
        return len(self.smiles)

    def __str__(self):
        return f"Dataset containing {len(self)} SMILES strings."

    @classmethod
    def collate_fn(cls, arr):
        """Receive a tensor of encoded sequences and dynamically pad to the longest sequence"""
        max_len = max([seq.size(0) for seq in arr])
        collated_tensor = torch.zeros(len(arr), max_len, dtype=torch.long)
        for i, seq in enumerate(arr):
            collated_tensor[i, :seq.size(0)] = seq
        return collated_tensor

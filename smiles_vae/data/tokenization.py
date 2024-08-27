import re

import numpy as np

SMI_REGEX_PATTERN = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"

class RegexTokenizer:
    """
    Class for tokenizing SMILES strings using a regular expression.
    Adapted from https://github.com/rxn4chemistry/rxnfp.

    Args:
        regex_pattern: regex pattern used for tokenization.
    """

    def __init__(self, regex_pattern=SMI_REGEX_PATTERN):
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, smiles):
        """
        Performs the regex tokenization.
        
        Args:
            smiles: smiles to tokenize.
        
        Returns:
            List of extracted tokens.
        """
        tokens = [token for token in self.regex.findall(smiles)]
        return tokens


class Vocabulary:
    """Class to encode/decode from SMILES to an array of indices"""
    def __init__(self,
                 init_from_file,
                 max_length=140,
                 ):
        self.tokens = []
        self.init_from_file(init_from_file)
        self.max_length = max_length

        self.vocab_size = len(self.tokens)
        self.vocab = dict(zip(self.tokens, range(len(self.tokens))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
    
    def init_from_file(self, file):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        with open(file, 'r') as f:
            tokens = f.read().split()
        self.tokens = [t for t in tokens]
    
    def tokenize(self, smi, add_bos=False, add_eos=False, add_pad=False):
        """Takes a single SMILES string and returns a list of tokens"""
        tokenizer = RegexTokenizer()
        tokenized_smi = tokenizer.tokenize(smi)

        if add_bos:
            tokenized_smi = ['<BOS>'] + tokenized_smi
        if add_eos:
            tokenized_smi = tokenized_smi + ['<EOS>']
        if add_pad:
            tokenized_smi = tokenized_smi + ['<PAD>']*(self.max_length - len(tokenized_smi))
        return tokenized_smi

    def encode(self, token_list):
        """
        Takes a list of tokens (e.g., '[NH]') and encodes to array of indices.
        If padding is desired, this should be done when tokenizing.
        """
        ids = [self.vocab[token] for token in token_list]
        return np.array(ids)
    
    def decode(self, ids, rem_bos=True, rem_eos=True, rem_pad=True):
        """Takes an array of indices and returns the corresponding SMILES string"""
        if len(ids) == 0:
            return ''
        
        if rem_bos and (ids[0] == self.vocab['<BOS>']):
            ids = ids[1:]
        
        tokens = []
        for i in ids:
            if rem_eos and (i == self.vocab['<EOS>']):
                break
            # pad tokens should typically only be present after the EOS token
            if rem_pad and (i == self.vocab['<PAD>']):
                break
            tokens.append(self.reversed_vocab[i])
        smi = "".join(tokens)
        return smi

    def __len__(self):
        return len(self.tokens)

    def __str__(self):
        return f"Vocabulary containing {len(self)} tokens: {self.tokens}"


def create_vocabulary(smiles_list, special_tokens=['<PAD>', '<BOS>', '<EOS>']):
    """Returns dictionary of all tokens present in a list of SMILES"""
    tokenizer = RegexTokenizer()
    vocab_set = set()
    for smi in smiles_list:
        token_list = tokenizer.tokenize(smi)
        vocab_set.update(token_list)
    
    # ensure that all numbers are in the vocabulary
    numbers = [str(i) for i in range(0, 10)]
    vocab_set.update(numbers)

    # sort the tokens for readability
    vocab_tokens = list(vocab_set)
    vocab_tokens.sort(key=lambda v: (v.upper(), v[0].islower()))

    # assemble vocabulary dictionary 
    vocab = {}
    for i, token in enumerate(special_tokens):
        vocab[token] = i
    
    _offset = len(vocab)
    for i, token in enumerate(vocab_tokens):
        vocab[token] = i + _offset
    
    return vocab

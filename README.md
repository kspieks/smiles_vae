# SMILES VAE
The goal of this repo is to utilize generative modeling techniques, namely VAE and conditional VAE, to create SMILES strings of molecules. 


## Pip installation instructions
As of April 2024, the [PyTorch](https://pytorch.org/get-started/locally/) website has the following statements:
- "PyTorch is supported on macOS 10.15 (Catalina) or above."
- "It is recommended that you use Python 3.8 - 3.11"

```
# create conda env
conda create -n smiles_vae python=3.11.8 -y

# activate conda env
conda activate smiles_vae

# install PyTorch for CPU only
pip install torch torchvision torchaudio

## install PyTorch for CUDA 11.8
## pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# install rdkit
pip install rdkit

# install other packages
pip install joblib jupyter pandarallel scikit-learn seaborn tqdm umap-learn

# install repo in editable mode
pip install -e .
```


# Other helpful resources
These examples provided inspiration for this code.

## Examples with GRU Encoder and GRU Decoder
https://github.com/molecularsets/moses/blob/master/moses/vae/model.py
https://github.com/bayeslabs/genmol/blob/master/genmol/vae/vae_model.py
https://github.com/wenhao-gao/mol_opt/blob/main/main/smiles_vae/models/smiles_vae/model.py

## Examples with Conv1D Encoder and GRU Decoder
Rafa's 2018 paper: https://pubs.acs.org/doi/10.1021/acscentsci.7b00572

These examples use TensorFlow code
https://github.com/aspuru-guzik-group/chemical_vae
https://github.com/aspuru-guzik-group/selfies/blob/master/examples/vae_example/chemistry_vae.py

https://github.com/znavoyan/vae-embeddings
https://github.com/YunjaeChoi/vaemols/blob/master/vaemols/models/vae.py


These examples use PyTorch
https://github.com/Ishan-Kumar2/Molecular_VAE_Pytorch/tree/master
https://github.com/aksub99/molecular-vae/blob/master/Molecular_VAE.ipynb
https://github.com/topazape/molecular-VAE
https://github.com/jessicaw9910/DL_Project/blob/main/src/model.py

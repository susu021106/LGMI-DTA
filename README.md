# LGMI-DTA

## 1. Conda environment

Use Linux or WSL Ubuntu with Conda. The versions below were recorded from the existing server environment; installation into a new environment has not been validated.

| Component | Version |
|---|---|
| Python | 3.9.23 |
| PyTorch | 2.3.1+cu121 |
| DGL | 0.9.1 (`dgl-cu113` build) |
| Other dependencies | See `requirements.txt` |

```bash
conda create -n lgmi-dta python=3.9.23 pip -c conda-forge -y
conda activate lgmi-dta
cd LGMI-DTA
python -m pip install -r requirements.txt
```

The requirements retain the server's legacy `dgl-cu113` build alongside PyTorch's CUDA 12.1 build. PyTorch wheels use the [official CUDA 12.1 index](https://pytorch.org/get-started/previous-versions/#v231); DGL wheels use the legacy wheel index specified in `requirements.txt`. Use an NVIDIA driver compatible with the installed GPU libraries for GPU training.

## 2. Prepare data and configuration

Before preprocessing, prepare the original fold CSVs, mapping tables, AlphaFold structures, pretrained ChemBERTa assets, and cached ESM embeddings:

```text
data/<dataset>/
├── <dataset>_compound_mapping.csv
├── <dataset>_protein_mapping.csv
├── DTA/fold/<fold>/<dataset>_train.csv
├── DTA/fold/<fold>/<dataset>_test.csv
├── PDB_AF2/AF-<protein_id>-F1-model_v4.pdb
└── processed/ESM_embedding_full/<protein_id>.npy
```

`<dataset>` is `Davis` or `KIBA`, and `<fold>` is 1–5. Fold CSVs must include `COMPOUND_ID`, `PROTEIN_ID`, `REG_LABEL`, and `RMSD`. The current scripts require these inputs to be prepared separately.

Before running the scripts:

1. Set `dataset` consistently in the active main blocks of `compound_process.py`, `protein_process.py`, and `train_DTA.py`. Preprocessing currently defaults to Davis; training defaults to KIBA.
2. In `compound_process.py`, replace `model_dir` with the local ChemBERTa directory containing its tokenizer, configuration, and weights.
3. For the first preprocessing run, uncomment the `Compound_graph_construction(...)` call if compound graph caches have not been generated.

## 3. Run in order

Run all commands from the project root with the Conda environment activated.

**Step 1: Generate compound features and fold files.**

```bash
python compound_process.py
```

**Step 2: Generate protein graphs and assemble protein fold files.**

```bash
python protein_process.py
```

**Step 3: Train and evaluate the full model.**

```bash
python train_DTA.py
```

The training entry point uses `MDGTDTInet` from `models/net.py`. Checkpoints are written under `results/<dataset>/fold_<fold>_seed<seed>/`. If all processed fold files already exist, start directly from Step 3.

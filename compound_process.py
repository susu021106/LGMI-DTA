import dgl
import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import Chem
from rdkit.Chem import rdmolfiles

from dgl.data.utils import save_graphs

from scipy import sparse as sp
from itertools import permutations
from scipy.spatial import distance_matrix
from dgl import load_graphs

import os
from transformers import AutoTokenizer, AutoConfig, AutoModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings

warnings.filterwarnings("ignore")

CHARISOSMISET = {
    "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
    "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
    "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
    "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
    "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
    "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
    "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
    "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64
}

def get_split_root(dataset, split_name):
    """
    split_name:
        warm
        cold_drug
        cold_protein
        cold_both
    """
    if split_name == 'warm':
        return os.path.join('data', dataset, 'DTA', 'fold')
    else:
        return os.path.join('data', dataset, 'DTA', split_name, 'fold')


def get_base_processed_root(dataset):
    """
    存放全局缓存：
        compound_graph/
        protein_graph/
        chemberta/
        ESM_embedding_full/
    """
    return os.path.join('data', dataset, 'processed')


def get_fold_processed_root(dataset, split_name):
    """
    存放当前划分的 fold 级别数据：
        processed/warm/train/fold/1/...
        processed/cold_drug/train/fold/1/...
    """
    return os.path.join('data', dataset, 'processed', split_name)


def make_fold_dirs(dir_output, fold):
    os.makedirs(os.path.join(dir_output, 'train', 'fold', str(fold)), exist_ok=True)
    os.makedirs(os.path.join(dir_output, 'test', 'fold', str(fold)), exist_ok=True)


def load_dataset_table(dataset):
    """
    直接从 data/<dataset>/<dataset>.csv 读取全量表。
    适配你现在的 KIBA.csv：
        COMPOUND_ID
        PROTEIN_ID
        COMPOUND_SMILES
        PROTEIN_SEQUENCE
        REG_LABEL
        RMSD
    """
    csv_path = os.path.join('data', dataset, f'{dataset}.csv')
    df = pd.read_csv(csv_path)
    df = df.dropna().reset_index(drop=True)

    required_cols = [
        'COMPOUND_ID',
        'PROTEIN_ID',
        'COMPOUND_SMILES',
        'PROTEIN_SEQUENCE',
        'REG_LABEL'
    ]

    for col in required_cols:
        assert col in df.columns, f'Missing column: {col}'

    if 'RMSD' not in df.columns:
        df['RMSD'] = 1

    return df

def normalize_compound_id(cid):
    """
    把 compound id 统一处理为字符串文件名。
    兼容数字型 ID 和 CHEMBLxxxx 这类字符串 ID。
    """
    cid = str(cid).strip()
    if len(cid) == 0:
        raise ValueError("Empty compound id encountered.")
    return cid


def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set {1}".format(x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def laplacian_positional_encoding(g, pos_enc_dim):
    """
    Graph positional encoding via Laplacian eigenvectors
    """
    A = g.adjacency_matrix(scipy_fmt='csr').astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    if EigVec.shape[1] < pos_enc_dim + 1:
        PadVec = np.zeros((EigVec.shape[0], pos_enc_dim + 1 - EigVec.shape[1]), dtype=EigVec.dtype)
        EigVec = np.concatenate((EigVec, PadVec), axis=1)

    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    return g


def atom_features(atom, explicit_H=False, use_chirality=True):
    """
    atom symbol(17), degree(7), formal charge(1), radical electrons(1),
    hybridization(6), aromatic(1), hydrogen atoms attached(5), chirality(3)
    """
    symbol = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'other']
    degree = [0, 1, 2, 3, 4, 5, 6]
    hybridizationType = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        'other'
    ]

    results = (
        one_of_k_encoding_unk(atom.GetSymbol(), symbol) +
        one_of_k_encoding(atom.GetDegree(), degree) +
        [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] +
        one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) +
        [atom.GetIsAromatic()]
    )

    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']
            ) + [atom.HasProp('_ChiralityPossible')]
        except Exception:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]

    return results


def bond_features(bond, use_chirality=True):
    """
    bond type(4), conjugated(1), in ring(1), stereo(4)
    """
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]

    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
        )
    return np.array(bond_feats).astype(int)


def smiles_to_graph(compound_id, smiles, explicit_H=False, use_chirality=True):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        raise RuntimeError(f"SMILES cannot be parsed for compound id: {compound_id}")

    if mol is None:
        raise RuntimeError(f"RDKit returned None for compound id: {compound_id}, smiles: {smiles}")

    g = dgl.DGLGraph()

    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    atom_feats = np.array([atom_features(a, explicit_H=explicit_H) for a in mol.GetAtoms()])

    if use_chirality:
        chiralcenters = Chem.FindMolChiralCenters(
            mol,
            force=True,
            includeUnassigned=True,
            useLegacyImplementation=False
        )
        chiral_arr = np.zeros([num_atoms, 3])
        for (i, rs) in chiralcenters:
            if rs == 'R':
                chiral_arr[i, 0] = 1
            elif rs == 'S':
                chiral_arr[i, 1] = 1
            else:
                chiral_arr[i, 2] = 1

        atom_feats = np.concatenate([atom_feats, chiral_arr], axis=1)

    g.ndata["atom"] = torch.tensor(atom_feats)

    # 保存原子索引和原子序数，方便解释性分析
    g.ndata["atom_idx"] = torch.arange(num_atoms, dtype=torch.long)
    g.ndata["atomic_num"] = torch.tensor(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()],
        dtype=torch.long
    )

    src_list = []
    dst_list = []
    bond_feats_all = []

    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()

        bond_feats = bond_features(bond, use_chirality=use_chirality)

        src_list.extend([u, v])
        dst_list.extend([v, u])
        bond_feats_all.append(bond_feats)
        bond_feats_all.append(bond_feats)

    g.add_edges(src_list, dst_list)
    g.edata["bond"] = torch.tensor(np.array(bond_feats_all))
    g = laplacian_positional_encoding(g, pos_enc_dim=8)
    return g


def load_chemberta_local(model_dir, device):
    os.environ["HF_HUB_OFFLINE"] = "1"

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True
    )

    config = AutoConfig.from_pretrained(
        model_dir,
        local_files_only=True
    )

    model = AutoModel.from_config(config)

    weight_path = os.path.join(model_dir, "pytorch_model.bin")
    state_dict = torch.load(weight_path, map_location="cpu")

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("lm_head."):
            continue
        if k.startswith("roberta."):
            k = k[len("roberta."):]
        new_state_dict[k] = v

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    print("ChemBERTa loaded from local bin.")
    print("hidden_size =", config.hidden_size)
    print("missing_keys =", len(missing_keys))
    print("unexpected_keys =", len(unexpected_keys))
    if len(missing_keys) > 0:
        print("missing_keys sample:", missing_keys[:10])
    if len(unexpected_keys) > 0:
        print("unexpected_keys sample:", unexpected_keys[:10])

    model = model.to(device)
    model.eval()

    return tokenizer, model, config


def ChemBERTa_embedding_construction(
        compound_ids,
        compound_smiles,
        save_dir,
        model_dir=r"D:\shy\ChemBERTa",
        max_length=128,
        device=None):
    """
    为每个 compound 的 SMILES 生成 ChemBERTa token-level embedding，
    保存为 .npy，形状 [L, 384]
    """
    os.makedirs(save_dir, exist_ok=True)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer, model, config = load_chemberta_local(model_dir, device)

    for idx, (cid, smi) in enumerate(zip(compound_ids, compound_smiles)):
        print(f'{idx + 1}/{len(compound_ids)}')

        cid = normalize_compound_id(cid)

        if not isinstance(smi, str) or len(smi.strip()) == 0:
            raise ValueError(f'Invalid SMILES for compound id {cid}: {smi}')

        save_path = os.path.join(save_dir, f'{cid}.npy')
        if os.path.exists(save_path):
            continue

        inputs = tokenizer(
            smi,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=False
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        embedding = outputs.last_hidden_state.squeeze(0)[1:-1].cpu().numpy().astype(np.float32)
        np.save(save_path, embedding)


def Compound_graph_construction(compound_ids, compound_values, dir_output):
    """
    为每个唯一 compound 生成单独图文件：
    data/<dataset>/processed/compound_graph/<compound_id>.bin
    """
    save_dir = os.path.join(dir_output, 'compound_graph')
    os.makedirs(save_dir, exist_ok=True)

    N = len(compound_values)
    for no, cid in enumerate(compound_ids):
        cid = normalize_compound_id(cid)
        print('/'.join(map(str, [no + 1, N])))

        smiles_data = compound_values[no]
        save_path = os.path.join(save_dir, f'{cid}.bin')
        if os.path.exists(save_path):
            continue

        compound_graph = smiles_to_graph(cid, smiles_data)
        dgl.save_graphs(save_path, [compound_graph])


def Compound_graph_process(dataset, fold, dir_output, id_train, id_test):
    compounds_graph_train, compounds_graph_test = [], []

    base_processed_root = get_base_processed_root(dataset)
    graph_cache_dir = os.path.join(base_processed_root, 'compound_graph')

    N = len(id_train)
    for no, cid in enumerate(id_train):
        cid = normalize_compound_id(cid)
        print(f'compound graph train {no + 1}/{N}')

        graph_path = os.path.join(graph_cache_dir, f'{cid}.bin')
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f'Missing compound graph: {graph_path}')

        compound_graph_train, _ = load_graphs(graph_path)
        compounds_graph_train.append(compound_graph_train[0])

    dgl.save_graphs(
        os.path.join(dir_output, 'train', 'fold', str(fold), 'compound_graph.bin'),
        compounds_graph_train
    )

    N = len(id_test)
    for no, cid in enumerate(id_test):
        cid = normalize_compound_id(cid)
        print(f'compound graph test {no + 1}/{N}')

        graph_path = os.path.join(graph_cache_dir, f'{cid}.bin')
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f'Missing compound graph: {graph_path}')

        compound_graph_test, _ = load_graphs(graph_path)
        compounds_graph_test.append(compound_graph_test[0])

    dgl.save_graphs(
        os.path.join(dir_output, 'test', 'fold', str(fold), 'compound_graph.bin'),
        compounds_graph_test
    )


def Compound_id_process(dataset, fold, dir_output, id_train, id_test):
    compounds_id_train, compounds_id_test = [], []

    N = len(id_train)
    for no, cid in enumerate(id_train):
        print('/'.join(map(str, [no + 1, N])))
        compounds_id_train.append(normalize_compound_id(cid))
    np.save(os.path.join(dir_output, 'train', 'fold', str(fold), 'compound_id.npy'), compounds_id_train)

    N = len(id_test)
    for no, cid in enumerate(id_test):
        print('/'.join(map(str, [no + 1, N])))
        compounds_id_test.append(normalize_compound_id(cid))
    np.save(os.path.join(dir_output, 'test', 'fold', str(fold), 'compound_id.npy'), compounds_id_test)


def Label_process(dataset, fold, dir_output, label_train, label_test):
    labels_train, labels_test = [], []

    N = len(label_train)
    for no, data in enumerate(label_train):
        print('/'.join(map(str, [no + 1, N])))
        labels_train.append(data)
    np.save(os.path.join(dir_output, 'train', 'fold', str(fold), 'label.npy'), labels_train)

    N = len(label_test)
    for no, data in enumerate(label_test):
        print('/'.join(map(str, [no + 1, N])))
        labels_test.append(data)
    np.save(os.path.join(dir_output, 'test', 'fold', str(fold), 'label.npy'), labels_test)


def Label_RMSD_process(dataset, fold, dir_output, label_train, label_test):
    labels_train, labels_test = [], []

    N = len(label_train)
    for no, data in enumerate(label_train):
        print('/'.join(map(str, [no + 1, N])))
        labels_train.append(data)
    np.save(os.path.join(dir_output, 'train', 'fold', str(fold), 'label_RMSD.npy'), labels_train)

    N = len(label_test)
    for no, data in enumerate(label_test):
        print('/'.join(map(str, [no + 1, N])))
        labels_test.append(data)
    np.save(os.path.join(dir_output, 'test', 'fold', str(fold), 'label_RMSD.npy'), labels_test)


def Compound_embedding_process(dataset, fold, dir_output, id_train, id_test):
    Compounds_embedding_train, Compounds_embedding_test = [], []

    base_processed_root = get_base_processed_root(dataset)
    emb_cache_dir = os.path.join(base_processed_root, 'chemberta')

    N = len(id_train)
    for no, cid in enumerate(id_train):
        cid = normalize_compound_id(cid)
        print(f'compound embedding train {no + 1}/{N}')

        emb_path = os.path.join(emb_cache_dir, f'{cid}.npy')
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f'Missing ChemBERTa embedding: {emb_path}')

        compound_embedding_train = np.load(emb_path, allow_pickle=True)
        Compounds_embedding_train.append(compound_embedding_train)

    Compounds_embedding_train = np.array(Compounds_embedding_train, dtype=object)
    np.save(
        os.path.join(dir_output, 'train', 'fold', str(fold), 'Compound_embedding.npy'),
        Compounds_embedding_train
    )

    N = len(id_test)
    for no, cid in enumerate(id_test):
        cid = normalize_compound_id(cid)
        print(f'compound embedding test {no + 1}/{N}')

        emb_path = os.path.join(emb_cache_dir, f'{cid}.npy')
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f'Missing ChemBERTa embedding: {emb_path}')

        compound_embedding_test = np.load(emb_path, allow_pickle=True)
        Compounds_embedding_test.append(compound_embedding_test)

    Compounds_embedding_test = np.array(Compounds_embedding_test, dtype=object)
    np.save(
        os.path.join(dir_output, 'test', 'fold', str(fold), 'Compound_embedding.npy'),
        Compounds_embedding_test
    )


if __name__ == '__main__':
    dataset = 'Davis'
    file_path = os.path.join('data', dataset, 'DTA', 'fold')
    file_path_compound = os.path.join('data', dataset, f'{dataset}_compound_mapping.csv')
    dir_output = os.path.join('data', dataset, 'processed')
    os.makedirs(dir_output, exist_ok=True)

    raw_data_compound = pd.read_csv(file_path_compound)
    compound_values = raw_data_compound['COMPOUND_SMILES'].values
    compound_id_unique = raw_data_compound['COMPOUND_ID'].values

    chemberta_dir = os.path.join(dir_output, 'chemberta')
    os.makedirs(chemberta_dir, exist_ok=True)

    # 1) 为每个唯一 compound 生成 ChemBERTa embedding
    ChemBERTa_embedding_construction(
        compound_ids=compound_id_unique,
        compound_smiles=compound_values,
        save_dir=chemberta_dir,
        model_dir=r"D:\shy\ChemBERTa",
        max_length=128
    )

    # # 2) 为每个唯一 compound 生成 graph
    # # 第一次跑建议打开；如果你已经生成过 compound_graph，可继续注释
    # Compound_graph_construction(
    #     compound_ids=compound_id_unique,
    #     compound_values=compound_values,
    #     dir_output=dir_output
    # )

    # 3) 处理 5 折
    for fold in range(1, 6):
        train_csv = os.path.join(file_path, str(fold), f'{dataset}_train.csv')
        test_csv = os.path.join(file_path, str(fold), f'{dataset}_test.csv')

        train_data = pd.read_csv(train_csv)
        train_data.dropna(inplace=True)

        test_data = pd.read_csv(test_csv)
        test_data.dropna(inplace=True)

        os.makedirs(os.path.join(dir_output, 'train', 'fold', str(fold)), exist_ok=True)
        os.makedirs(os.path.join(dir_output, 'test', 'fold', str(fold)), exist_ok=True)

        compound_id_train = train_data['COMPOUND_ID'].values
        compound_id_test = test_data['COMPOUND_ID'].values

        label_train = train_data['REG_LABEL'].values
        label_test = test_data['REG_LABEL'].values
        label_RMSD_train = train_data['RMSD'].values
        label_RMSD_test = test_data['RMSD'].values

        Compound_graph_process(
            dataset=dataset,
            fold=fold,
            id_train=compound_id_train,
            id_test=compound_id_test,
            dir_output=dir_output
        )

        Compound_id_process(
            dataset=dataset,
            fold=fold,
            id_train=compound_id_train,
            id_test=compound_id_test,
            dir_output=dir_output
        )

        Label_process(
            dataset=dataset,
            fold=fold,
            dir_output=dir_output,
            label_train=label_train,
            label_test=label_test
        )

        Label_RMSD_process(
            dataset=dataset,
            fold=fold,
            dir_output=dir_output,
            label_train=label_RMSD_train,
            label_test=label_RMSD_test
        )

        Compound_embedding_process(
            dataset=dataset,
            fold=fold,
            id_train=compound_id_train,
            id_test=compound_id_test,
            dir_output=dir_output
        )

    print('The preprocess of ' + dataset + ' dataset has finished!')


# if __name__ == '__main__':
#     dataset = 'Davis'
#
#     # 可选：
#     # warm
#     # cold_drug
#     # cold_protein
#     # cold_both
#     split_name = 'cold_both'
#
#     split_root = get_split_root(dataset, split_name)
#     base_processed_root = get_base_processed_root(dataset)
#     dir_output = get_fold_processed_root(dataset, split_name)
#
#     os.makedirs(base_processed_root, exist_ok=True)
#     os.makedirs(dir_output, exist_ok=True)
#
#     raw_data = load_dataset_table(dataset)
#
#     compound_id_unique = raw_data['COMPOUND_ID'].astype(str).drop_duplicates().values
#     compound_smiles_unique = (
#         raw_data[['COMPOUND_ID', 'COMPOUND_SMILES']]
#         .astype({'COMPOUND_ID': str})
#         .drop_duplicates(subset=['COMPOUND_ID'])
#         ['COMPOUND_SMILES']
#         .values
#     )
#
#     chemberta_dir = os.path.join(base_processed_root, 'chemberta')
#     os.makedirs(chemberta_dir, exist_ok=True)
#
#     # # 1. 生成所有 compound 的 ChemBERTa embedding
#     # # 已经生成过可以注释掉
#     # ChemBERTa_embedding_construction(
#     #     compound_ids=compound_id_unique,
#     #     compound_smiles=compound_smiles_unique,
#     #     save_dir=chemberta_dir,
#     #     model_dir=r"D:\shy\ChemBERTa",
#     #     max_length=128
#     # )
#
#     # # 2. 生成所有 compound graph
#     # # 已经生成过可以注释掉
#     # Compound_graph_construction(
#     #     compound_ids=compound_id_unique,
#     #     compound_values=compound_smiles_unique,
#     #     dir_output=base_processed_root
#     # )
#
#     # 3. 处理当前 split 的 5 fold
#     for fold in range(1, 6):
#         print('=' * 80)
#         print(f'Processing {dataset}, split={split_name}, fold={fold}')
#         print('=' * 80)
#
#         make_fold_dirs(dir_output, fold)
#
#         train_csv = os.path.join(split_root, str(fold), f'{dataset}_train.csv')
#         test_csv = os.path.join(split_root, str(fold), f'{dataset}_test.csv')
#
#         train_data = pd.read_csv(train_csv).dropna().reset_index(drop=True)
#         test_data = pd.read_csv(test_csv).dropna().reset_index(drop=True)
#
#         compound_id_train = train_data['COMPOUND_ID'].astype(str).values
#         compound_id_test = test_data['COMPOUND_ID'].astype(str).values
#
#         label_train = train_data['REG_LABEL'].values
#         label_test = test_data['REG_LABEL'].values
#
#         if 'RMSD' in train_data.columns:
#             label_RMSD_train = train_data['RMSD'].values
#         else:
#             label_RMSD_train = np.ones(len(train_data))
#
#         if 'RMSD' in test_data.columns:
#             label_RMSD_test = test_data['RMSD'].values
#         else:
#             label_RMSD_test = np.ones(len(test_data))
#
#         Compound_graph_process(
#             dataset=dataset,
#             fold=fold,
#             id_train=compound_id_train,
#             id_test=compound_id_test,
#             dir_output=dir_output
#         )
#
#         Compound_id_process(
#             dataset=dataset,
#             fold=fold,
#             id_train=compound_id_train,
#             id_test=compound_id_test,
#             dir_output=dir_output
#         )
#
#         Label_process(
#             dataset=dataset,
#             fold=fold,
#             dir_output=dir_output,
#             label_train=label_train,
#             label_test=label_test
#         )
#
#         Label_RMSD_process(
#             dataset=dataset,
#             fold=fold,
#             dir_output=dir_output,
#             label_train=label_RMSD_train,
#             label_test=label_RMSD_test
#         )
#
#         Compound_embedding_process(
#             dataset=dataset,
#             fold=fold,
#             id_train=compound_id_train,
#             id_test=compound_id_test,
#             dir_output=dir_output
#         )
#
#     print(f'The compound preprocess of {dataset}, split={split_name} has finished!')
import os
import pickle
import timeit

import deepchem
import numpy as np
import pandas as pd
import torch
import dgl
from rdkit import Chem
from scipy import sparse as sp
import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals
from MDAnalysis.analysis import distances
from itertools import product, groupby, permutations
from scipy.spatial import distance_matrix
from dgl import load_graphs
import warnings

print("cwd =", os.getcwd())

warnings.filterwarnings("ignore")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device('cuda')

METAL = ["LI", "NA", "K", "RB", "CS", "MG", "TL", "CU", "AG", "BE", "NI", "PT", "ZN", "CO", "PD", "AG", "CR", "FE", "V",
         "MN", "HG", 'GA',
         "CD", "YB", "CA", "SN", "PB", "EU", "SR", "SM", "BA", "RA", "AL", "IN", "TL", "Y", "LA", "CE", "PR", "ND",
         "GD", "TB", "DY", "ER",
         "TM", "LU", "HF", "ZR", "CE", "U", "PU", "TH"]
RES_MAX_NATOMS = 24


CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

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

def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def obtain_self_dist(res):
    try:
        # xx = res.atoms.select_atoms("not name H*")
        xx = res.atoms
        dists = distances.self_distance_array(xx.positions)
        ca = xx.select_atoms("name CA")
        c = xx.select_atoms("name C")
        n = xx.select_atoms("name N")
        o = xx.select_atoms("name O")
        return [dists.max() * 0.1, dists.min() * 0.1, distances.dist(ca, o)[-1][0] * 0.1,
                distances.dist(o, n)[-1][0] * 0.1, distances.dist(n, c)[-1][0] * 0.1]
    except:
        return [0, 0, 0, 0, 0]


def obtain_dihediral_angles(res):
    try:
        if res.phi_selection() is not None:
            phi = res.phi_selection().dihedral.value()
        else:
            phi = 0
        if res.psi_selection() is not None:
            psi = res.psi_selection().dihedral.value()
        else:
            psi = 0
        if res.omega_selection() is not None:
            omega = res.omega_selection().dihedral.value()
        else:
            omega = 0
        if res.chi1_selection() is not None:
            chi1 = res.chi1_selection().dihedral.value()
        else:
            chi1 = 0
        return [phi * 0.01, psi * 0.01, omega * 0.01, chi1 * 0.01]
    except:
        return [0, 0, 0, 0]


def calc_res_features(res):
    return np.array(one_of_k_encoding_unk(obtain_resname(res),
                                          ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR',
                                           'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP',
                                           'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
                                           'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP', 'M', 'X']) +  # 32  residue type
					obtain_self_dist(res) +  # 5
					obtain_dihediral_angles(res)  # 4
					)


def obtain_resname(res):
    if res.resname[:2] == "CA":
        resname = "CA"
    elif res.resname[:2] == "FE":
        resname = "FE"
    elif res.resname[:2] == "CU":
        resname = "CU"
    else:
        resname = res.resname.strip()

    if resname in METAL:
        return "M"
    else:
        return resname


##'FE', 'SR', 'GA', 'IN', 'ZN', 'CU', 'MN', 'SR', 'K' ,'NI', 'NA', 'CD' 'MG','CO','HG', 'CS', 'CA',

#
def obatin_edge(u, cutoff=10.0):
    edgeids = []
    dismin = []
    dismax = []
    # permutations(u.residues, 2)遍历u中所有两两组合，12 13 23 21 31 32，，每一对残疾
    for res1, res2 in permutations(u.residues, 2):
        # 两个残迹间左右原子的距离返回，所有点对点距离，n*m矩阵，元素距离
        dist = calc_dist(res1, res2)
        # 最小距离小于cutoff,残基对索引,加入edgeids,最小最大同时，res1.ix, res2.ix索引
        if dist.min() <= cutoff:
            edgeids.append([res1.ix, res2.ix])
            dismin.append(dist.min() * 0.1)
            dismax.append(dist.max() * 0.1)
    return edgeids, np.array([dismin, dismax]).T


def check_connect(u, i, j):
    if abs(i - j) != 1:
        return 0
    else:
        if i > j:
            i = j
        nb1 = len(u.residues[i].get_connections("bonds"))
        nb2 = len(u.residues[i + 1].get_connections("bonds"))
        nb3 = len(u.residues[i:i + 2].get_connections("bonds"))
        if nb1 + nb2 == nb3 + 1:
            return 1
        else:
            return 0


# 两个残迹间左右原子的距离返回，所有点对点距离，n*m矩阵，元素距离
def calc_dist(res1, res2):

    # res1.atoms.positions，mdanalysis库中属性，包含原子3d坐标numpy数组，每行代表一个原子坐标
    dist_array = distances.distance_array(res1.atoms.positions, res2.atoms.positions)
    return dist_array


def load_protein(protpath, explicit_H=False, use_chirality=True):

    mol = Chem.MolFromPDBFile(protpath, sanitize=True, removeHs=True, flavor=0, proximityBonding=False)
    if use_chirality:
        Chem.AssignStereochemistryFrom3D(mol)
    return mol


def prot_to_graph(id, prot_pdb, cutoff=10.0):
    # 从数据集拿一个蛋白质id，一个对应PDB
    # 使用DeepChem库中的ConvexHullPocketFinder类创建一个对象pk，用于找到蛋白质的凸包口袋。
    pk = deepchem.dock.ConvexHullPocketFinder()
    #  使用RDKit库中的Chem模块从PDB文件中读取蛋白质结构，并将其转换为分子对象prot。
    prot = Chem.MolFromPDBFile(prot_pdb, sanitize=True, removeHs=True, flavor=0, proximityBonding=False)
    # 使用RDKit库中的Chem模块为prot分子对象分配立体化学配置。
    Chem.AssignStereochemistryFrom3D(prot)
    # 使用MDAnalysis库创建一个Universe对象u，它代表了整个分子系统。
    u = mda.Universe(prot)
    # 使用DGL库创建一个空的图对象g。
    g = dgl.DGLGraph()


    # Add nodes
    # 计算Universe对象u中的残基数量，并将结果存储在num_residues中。
    num_residues = len(u.residues)
    # 在图g中添加num_residues个节点。
    g.add_nodes(num_residues)
    # 对Universe对象u中的每个残基调用++calc_res_features函数（22+5+4维）计算特征，并将结果存储在NumPy数组res_feats中。
    res_feats = np.array([calc_res_features(res) for res in u.residues])

    # # 从.npy文件中加载ESM嵌入特征，并将结果存储在esm_feats中。
    # esm_feats = np.load('data/Davis/processed/ESM_embedding/' + id + '.npy', allow_pickle=True)
    # # 计算esm_feats的长度，并将结果存储在len_esm中。esm_feats行数
    # len_esm = np.size(esm_feats, 0)
    # # 如果esm_feats的长度小于num_residues，则使用0填充esm_feats，使其长度等于num_residues。如果esm_feats的长度大于num_residues，则截取esm_feats的前num_residues个元素。 残基数量
    # if len_esm < num_residues:
    #     esm_feats = np.pad(esm_feats, ((0, num_residues-len_esm), (0,0)), 'constant')
    # elif len_esm > num_residues:
    #     esm_feats = esm_feats[:num_residues, :]
    #
    # #  将res_feats和esm_feats沿着列方向（axis=1）连接起来，形成蛋白质特征prot_feats。左右链接
    # prot_feats = np.concatenate((res_feats, esm_feats), axis=1)

    prot_feats =res_feats
    #  将prot_feats转换为PyTorch张量，并将其作为节点数据添加到图g中。
    g.ndata["feats"] = torch.tensor(prot_feats)

    # 保存 residue 原始索引和 PDB residue id，方便解释性分析
    g.ndata["residue_ix"] = torch.tensor(
        [res.ix for res in u.residues],
        dtype=torch.long
    )

    g.ndata["residue_id"] = torch.tensor(
        [res.resid for res in u.residues],
        dtype=torch.long
    )

    #  调用++obatin_edge函数(最小距离残疾对)获取边的ID和距离矩阵。edgeids两个残疾最小原子距离小于的残疾索引对列表，distm两列，第一列最小第二列最大距离
    edgeids, distm = obatin_edge(u, cutoff)
    # 将边的ID解压缩为源节点列表src_list和目标节点列表dst_list。src_list第一列，dst_list第二列
    src_list, dst_list = zip(*edgeids)
    # 在图g中添加边。
    g.add_edges(src_list, dst_list)


    # 对Universe对象u中的每个残基调用obtain_ca_pos函数获取Cα原子
    # (在蛋白质中，Cα原子是主链骨架的组成原子。每个氨基酸残基都有一个Cα原子，它连接着氨基和羧基，以及氨基酸的侧链。因此，Cα原子的位置和排列顺序决定了蛋白质的三维结构)的位置，并将结果作为节点数据添加到图g中。
    # m残疾第一个原子，ca位置，没有ca返回平均位置
    g.ndata["ca_pos"] = torch.tensor(np.array([obtain_ca_pos(res) for res in u.residues]))

    # 计算Universe对象u中的每个残基的质心位置，并将结果作为节点数据添加到图g中。
    g.ndata["center_pos"] = torch.tensor(u.atoms.center_of_mass(compound='residues'))

    # 计算Cα原子位置的距离矩阵。ca原子两两之间的距离 对称阵
    dis_matx_ca = distance_matrix(g.ndata["ca_pos"], g.ndata["ca_pos"])
    # 计算边的Cα距离，并将结果乘以0.1。cadist是一个包含了edgeids中指定的残基对之间距离（缩小10倍）的PyTorch张量
    cadist = torch.tensor([dis_matx_ca[i, j] for i, j in edgeids]) * 0.1
    # 计算质心位置的距离矩阵。
    dis_matx_center = distance_matrix(g.ndata["center_pos"], g.ndata["center_pos"])
    # 计算边的质心距离，并将结果乘以0.1。
    cedist = torch.tensor([dis_matx_center[i, j] for i, j in edgeids]) * 0.1

    # 检查边是否连接，并将结果转换为PyTorch张量。前面距离小于设定集
    edge_connect = torch.tensor(np.array([check_connect(u, x, y) for x, y in zip(src_list, dst_list)]))
    # 将边的连接性、Cα距离、质心距离和距离矩阵连接起来，形成边的特征，并将结果作为边数据添加到图g中。
    g.edata["feats"] = torch.cat([edge_connect.view(-1, 1), cadist.view(-1, 1), cedist.view(-1, 1), torch.tensor(distm)], dim=1)
    # 从图g的节点数据中删除Cα原子位置和质心位置。
    g.ndata.pop("ca_pos")
    g.ndata.pop("center_pos")

    # 对Universe对象u中的每个残基调用obtain_ca_pos函数获取Cα原子的位置，并将结果存储在ca_pos中。
    ca_pos = np.array(np.array([obtain_ca_pos(res) for res in u.residues]))
    # 使用ConvexHullPocketFinder对象pk找到蛋白质的凸包口袋，并将结果存储在pockets中。
    # 返回的pockets是一个列表，其中每个元素都是一个CoordinateBox对象，CoordinateBox是DeepChem库中的一个类，它用于表示三维空间中的一个立方体区域
    # CoordinateBox对象由三个元素定义：x轴的范围、y轴的范围和z轴的范围1。每个范围都是一个元组，包含了该轴的最小值和最大值
    pockets = pk.find_pockets(prot_pdb)
    # 对每个凸包口袋进行遍历，如果Cα原子的位置在凸包口袋的范围内，则将其索引添加到idxs中。(找出位于蛋白质口袋中的Cα原子)
    for bound_box in pockets:
        x_min = bound_box.x_range[0]
        x_max = bound_box.x_range[1]
        y_min = bound_box.y_range[0]
        y_max = bound_box.y_range[1]
        z_min = bound_box.z_range[0]
        z_max = bound_box.z_range[1]
        idxs = []
        for idx in range(ca_pos.shape[0]):
            if x_min < ca_pos[idx][0] < x_max and y_min < ca_pos[idx][1] < y_max and z_min < ca_pos[idx][2] < z_max:
                idxs.append(idx)

    #  从图g中提取出idxs中的节点，形成子图g_pocket。
    g_pocket = dgl.node_subgraph(g, idxs)
    # 对子图g_pocket进行拉普拉斯位置编码。
    g_pocket = laplacian_positional_encoding(g_pocket, pos_enc_dim=8)
    return g_pocket


def obtain_ca_pos(res):
    # 残基名称是否为"M"
    if obtain_resname(res) == "M":
        # 返回第一个原子位置
        return res.atoms.positions[0]
    else:
        try:
            # ca位置
            pos = res.atoms.select_atoms("name CA").positions[0]
            return pos
        except:  ##some residues loss the CA atoms
            # 没有ca，原子平均值
            return res.atoms.positions.mean(axis=0)


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix(scipy_fmt='csr').astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    if EigVec.shape[1] < pos_enc_dim + 1:
        PadVec = np.zeros((EigVec.shape[0], pos_enc_dim + 1 - EigVec.shape[1]), dtype=EigVec.dtype)
        EigVec = np.concatenate((EigVec, PadVec), 1)
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    return g

# 所有蛋白质graph
def Protein_graph_construction(dataset, protein_ids, dir_output):
    N = len(protein_ids)
    save_dir = os.path.join(dir_output, 'protein_graph')
    os.makedirs(save_dir, exist_ok=True)

    for no, pid in enumerate(protein_ids):
        pid = str(pid).strip()
        print(f'protein graph construction {no + 1}/{N}')

        save_path = os.path.join(save_dir, f'{pid}.bin')
        if os.path.exists(save_path):
            continue

        pdb_path = os.path.join(
            'data',
            dataset,
            'PDB_AF2',
            f'AF-{pid}-F1-model_v4.pdb'
        )

        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f'Missing PDB file: {pdb_path}')

        protein_graph = prot_to_graph(pid, pdb_path, cutoff=10.0)
        dgl.save_graphs(save_path, [protein_graph])


# davis 1 蛋白质ID，Q99755..  train test process
def Protein_graph_process(dataset, fold, dir_output, id_train, id_test):
    proteins_graph_train, proteins_graph_test = [], []

    base_processed_root = get_base_processed_root(dataset)
    graph_cache_dir = os.path.join(base_processed_root, 'protein_graph')

    N = len(id_train)
    for no, pid in enumerate(id_train):
        pid = str(pid).strip()
        print(f'protein graph train {no + 1}/{N}')

        graph_path = os.path.join(graph_cache_dir, f'{pid}.bin')
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f'Missing protein graph: {graph_path}')

        protein_graph_train, _ = load_graphs(graph_path)
        proteins_graph_train.append(protein_graph_train[0])

    dgl.save_graphs(
        os.path.join(dir_output, 'train', 'fold', str(fold), 'protein_graph.bin'),
        proteins_graph_train
    )

    N = len(id_test)
    for no, pid in enumerate(id_test):
        pid = str(pid).strip()
        print(f'protein graph test {no + 1}/{N}')

        graph_path = os.path.join(graph_cache_dir, f'{pid}.bin')
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f'Missing protein graph: {graph_path}')

        protein_graph_test, _ = load_graphs(graph_path)
        proteins_graph_test.append(protein_graph_test[0])

    dgl.save_graphs(
        os.path.join(dir_output, 'test', 'fold', str(fold), 'protein_graph.bin'),
        proteins_graph_test
    )


def Protein_embedding_process(dataset, fold, dir_output, id_train, id_test):
    proteins_embedding_train, proteins_embedding_test = [], []

    base_processed_root = get_base_processed_root(dataset)
    emb_cache_dir = os.path.join(base_processed_root, 'ESM_embedding_full')

    N = len(id_train)
    for no, pid in enumerate(id_train):
        pid = str(pid).strip()
        print(f'protein embedding train {no + 1}/{N}')

        emb_path = os.path.join(emb_cache_dir, f'{pid}.npy')
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f'Missing ESM embedding: {emb_path}')

        protein_embedding_train = np.load(emb_path, allow_pickle=True)
        proteins_embedding_train.append(protein_embedding_train)

    proteins_embedding_train = np.array(proteins_embedding_train, dtype=object)
    np.save(
        os.path.join(dir_output, 'train', 'fold', str(fold), 'protein_embedding.npy'),
        proteins_embedding_train
    )

    N = len(id_test)
    for no, pid in enumerate(id_test):
        pid = str(pid).strip()
        print(f'protein embedding test {no + 1}/{N}')

        emb_path = os.path.join(emb_cache_dir, f'{pid}.npy')
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f'Missing ESM embedding: {emb_path}')

        protein_embedding_test = np.load(emb_path, allow_pickle=True)
        proteins_embedding_test.append(protein_embedding_test)

    proteins_embedding_test = np.array(proteins_embedding_test, dtype=object)
    np.save(
        os.path.join(dir_output, 'test', 'fold', str(fold), 'protein_embedding.npy'),
        proteins_embedding_test
    )


def Protein_id_process(dataset, fold, dir_output, id_train, id_test):
    proteins_id_train, proteins_id_test = [], []

    N = len(id_train)
    for no, pid in enumerate(id_train):
        print(f'protein id train {no + 1}/{N}')
        proteins_id_train.append(str(pid).strip())

    np.save(
        os.path.join(dir_output, 'train', 'fold', str(fold), 'protein_id.npy'),
        proteins_id_train
    )

    N = len(id_test)
    for no, pid in enumerate(id_test):
        print(f'protein id test {no + 1}/{N}')
        proteins_id_test.append(str(pid).strip())

    np.save(
        os.path.join(dir_output, 'test', 'fold', str(fold), 'protein_id.npy'),
        proteins_id_test
    )


if __name__ == '__main__':
    dataset = 'Davis'

    # 原始 warm split 路径
    file_path = os.path.join('data', dataset, 'DTA', 'fold')

    # protein mapping 文件
    file_path_protein = os.path.join(
        'data',
        dataset,
        f'{dataset}_protein_mapping.csv'
    )

    # 输出目录：仍然是原来的 data/Davis/processed/
    dir_output = os.path.join('data', dataset, 'processed')
    os.makedirs(dir_output, exist_ok=True)

    # 创建 protein_graph 全局缓存目录
    os.makedirs(os.path.join(dir_output, 'protein_graph'), exist_ok=True)

    # --------------------------------------------------
    # 1. 读取全部唯一 protein id
    # --------------------------------------------------
    raw_data_protein = pd.read_csv(file_path_protein)

    protein_id_unique = (
        raw_data_protein['PROTEIN_ID']
        .astype(str)
        .str.strip()
        .unique()
    )

    print(f'Total unique proteins: {len(protein_id_unique)}')

    # --------------------------------------------------
    # 2. 构建所有 protein graph
    #    保存到：
    #    data/Davis/processed/protein_graph/<protein_id>.bin
    # --------------------------------------------------
    Protein_graph_construction(
        dataset=dataset,
        protein_ids=protein_id_unique,
        dir_output=dir_output
    )

    # --------------------------------------------------
    # 3. 按 5-fold 保存 train/test 的 protein graph、embedding、id
    # --------------------------------------------------
    for fold in range(1, 6):
        print('\n' + '=' * 80)
        print(f'Processing {dataset}, fold {fold}')
        print('=' * 80)

        # 当前 fold 的输出目录
        os.makedirs(
            os.path.join(dir_output, 'train', 'fold', str(fold)),
            exist_ok=True
        )
        os.makedirs(
            os.path.join(dir_output, 'test', 'fold', str(fold)),
            exist_ok=True
        )

        train_csv = os.path.join(
            file_path,
            str(fold),
            f'{dataset}_train.csv'
        )

        test_csv = os.path.join(
            file_path,
            str(fold),
            f'{dataset}_test.csv'
        )

        if not os.path.exists(train_csv):
            raise FileNotFoundError(f'Missing train csv: {train_csv}')

        if not os.path.exists(test_csv):
            raise FileNotFoundError(f'Missing test csv: {test_csv}')

        train_data = pd.read_csv(train_csv)
        train_data.dropna(inplace=True)

        test_data = pd.read_csv(test_csv)
        test_data.dropna(inplace=True)

        protein_id_train = (
            train_data['PROTEIN_ID']
            .astype(str)
            .str.strip()
            .values
        )

        protein_id_test = (
            test_data['PROTEIN_ID']
            .astype(str)
            .str.strip()
            .values
        )

        # --------------------------------------------------
        # 保存 protein graph
        # data/Davis/processed/train/fold/1/protein_graph.bin
        # data/Davis/processed/test/fold/1/protein_graph.bin
        # --------------------------------------------------
        Protein_graph_process(
            dataset=dataset,
            fold=fold,
            dir_output=dir_output,
            id_train=protein_id_train,
            id_test=protein_id_test
        )

        # --------------------------------------------------
        # 保存 protein embedding
        # data/Davis/processed/train/fold/1/protein_embedding.npy
        # data/Davis/processed/test/fold/1/protein_embedding.npy
        # --------------------------------------------------
        Protein_embedding_process(
            dataset=dataset,
            fold=fold,
            dir_output=dir_output,
            id_train=protein_id_train,
            id_test=protein_id_test
        )

        # --------------------------------------------------
        # 保存 protein id
        # data/Davis/processed/train/fold/1/protein_id.npy
        # data/Davis/processed/test/fold/1/protein_id.npy
        # --------------------------------------------------
        Protein_id_process(
            dataset=dataset,
            fold=fold,
            dir_output=dir_output,
            id_train=protein_id_train,
            id_test=protein_id_test
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
#     protein_id_unique = (
#         raw_data['PROTEIN_ID']
#         .astype(str)
#         .drop_duplicates()
#         .values
#     )
#
#     # # 1. 生成所有 protein graph
#     # # 已经生成过可以注释掉
#     # Protein_graph_construction(
#     #     dataset=dataset,
#     #     protein_ids=protein_id_unique,
#     #     dir_output=base_processed_root
#     # )
#
#     # 2. 处理当前 split 的 5 fold
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
#         protein_id_train = train_data['PROTEIN_ID'].astype(str).values
#         protein_id_test = test_data['PROTEIN_ID'].astype(str).values
#
#         Protein_graph_process(
#             dataset=dataset,
#             fold=fold,
#             id_train=protein_id_train,
#             id_test=protein_id_test,
#             dir_output=dir_output
#         )
#
#         Protein_embedding_process(
#             dataset=dataset,
#             fold=fold,
#             id_train=protein_id_train,
#             id_test=protein_id_test,
#             dir_output=dir_output
#         )
#
#         Protein_id_process(
#             dataset=dataset,
#             fold=fold,
#             id_train=protein_id_train,
#             id_test=protein_id_test,
#             dir_output=dir_output
#         )
#
#     print(f'The protein preprocess of {dataset}, split={split_name} has finished!')

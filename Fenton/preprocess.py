import pandas as pd
import numpy as np
import globalvaribale
import copy
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm2D import Generate
from rdkit import Chem
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.rdmolops import PatternFingerprint
import os
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import rdMolDescriptors


def data_load():
    path = globalvaribale.get_value('PATH')
    data = pd.read_csv(path).values[:, 1:]
    return data


def delete_nan_mask(data):
    data = data.astype(np.str)
    mask = []
    for index in range(data.shape[0]):
        element = data[index, :]
        if "nan" in element:
            mask.append(False)
        else:
            mask.append(True)
    return mask


def rdkit(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return list(Chem.RDKFingerprint(mol))


def atom_pairs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return list(Pairs.GetAtomPairFingerprintAsBitVect(mol))


def bag_of_topological_torsions():
    map_pollutant_to_smiles = globalvaribale.get_value("MAP_POLLUTANT_TO_SMILES")
    smiles_list = [item[1] for item in map_pollutant_to_smiles.items()]
    bag_list = []
    for smiles_temp in smiles_list:
        mol = Chem.MolFromSmiles(smiles_temp)
        fp_list = []
        fp = Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol)
        dict_fp = fp.GetNonzeroElements()
        for item in dict_fp.keys():
            fp_list.append(item)
        bag_list += fp_list
    bag_list = list(set(bag_list))
    bag_list.sort()
    return bag_list


def topological_torsions(smiles):
    bag_list = bag_of_topological_torsions()
    mol = Chem.MolFromSmiles(smiles)
    fingerprint_list = []
    fingerprint = Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol)
    fingerprint_dict = fingerprint.GetNonzeroElements()
    for item in fingerprint_dict.keys():
        fingerprint_list.append(item)
    fingerprint_list = [1 if element in fingerprint_list else 0 for element in bag_list]
    return fingerprint_list


def maccs_key(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return list(MACCSkeys.GenMACCSKeys(mol))


def morgan(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    return list(GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))


def pharmacophore(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    feat_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    from rdkit.Chem.Pharm2D.SigFactory import SigFactory
    sig_factory = SigFactory(feat_factory, minPointCount=2, maxPointCount=3)
    sig_factory.SetBins([(0, 1), (1, 3), (3, 8)])
    sig_factory.Init()
    fp = Generate.Gen2DFingerprint(mol, sig_factory)
    return list(fp)


def pattern(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return list(PatternFingerprint(mol))


def one_hot(smiles):
    map_pollutant_to_smiles = globalvaribale.get_value("MAP_POLLUTANT_TO_SMILES")
    smiles_list = [item[1] for item in map_pollutant_to_smiles.items()]
    zeros_temp = np.zeros(len(smiles_list)).tolist()
    position_smiles = smiles_list.index(smiles)
    zeros_temp[position_smiles] = 1
    return zeros_temp


def dragon_descriptors(smiles):
    structural_descriptors_database = globalvaribale.get_value('structural_descriptors_path')
    smiles_list = structural_descriptors_database[1:, 0].tolist()
    database = structural_descriptors_database[1:, 1:]
    position_smiles = smiles_list.index(smiles)
    values = database[position_smiles, :].tolist()
    return values


def Morse(smi):
    mol = AllChem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=0)
        AllChem.MMFFOptimizeMolecule(mol)
        des = rdMolDescriptors.CalcMORSE(mol)
    except:
        des = []
    return des


def WHIM(smi):
    mol = AllChem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=0)
        AllChem.MMFFOptimizeMolecule(mol)
        des = rdMolDescriptors.CalcWHIM(mol)
    except:
        des = []
    return des


def RDKit_Morse(smi):
    mol = AllChem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=0)
        AllChem.MMFFOptimizeMolecule(mol)
        des = rdMolDescriptors.CalcMORSE(mol)
    except:
        des = []

    return des + list(Chem.RDKFingerprint(mol))


def Morgan_Morse(smi):
    mol = AllChem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=0)
        AllChem.MMFFOptimizeMolecule(mol)
        des = rdMolDescriptors.CalcMORSE(mol)
    except:
        des = []

    return des + list(GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))


def RDKit_WHIM(smi):
    mol = AllChem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=0)
        AllChem.MMFFOptimizeMolecule(mol)
        des = rdMolDescriptors.CalcWHIM(mol)
    except:
        des = []

    return des + list(Chem.RDKFingerprint(mol))


def Morgan_WHIM(smi):
    mol = AllChem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=0)
        AllChem.MMFFOptimizeMolecule(mol)
        des = rdMolDescriptors.CalcWHIM(mol)
    except:
        des = []

    # if j == 8:
    #     from rdkit.Chem import Draw
    #     from rdkit.Chem.Draw import IPythonConsole
    #     drawOptions = Draw.rdMolDraw2D.MolDrawOptions()
    #     drawOptions.prepareMolsBeforeDrawing = False
    #     ECFP_bitinfo = {}
    #     ECFP = AllChem.GetMorganFingerprint(mol, radius=2, bitInfo=ECFP_bitinfo, useFeatures=False)
    #     ECFP_tuples = [(mol, bit, ECFP_bitinfo) for bit in list(ECFP_bitinfo.keys())]
    #     img = Draw.DrawMorganBits(ECFP_tuples, molsPerRow=5, legends=list(map(str, list(ECFP_bitinfo.keys()))),
    #                               )
    #     img.save(str(j)+'.jpg')


    # from rdkit.Chem import Draw
    # from rdkit.Chem.Draw import IPythonConsole
    # drawOptions = Draw.rdMolDraw2D.MolDrawOptions()
    # drawOptions.prepareMolsBeforeDrawing = False
    # bi = {}
    # fp = GetMorganFingerprintAsBitVect(Chem.AddHs(Chem.MolFromSmiles(smi)), radius=2, bitInfo=bi, nBits=2048)
    # tpls = [(Chem.AddHs(Chem.MolFromSmiles(smi)), x, bi) for x in fp.GetOnBits()]
    # if j != 15:
    #     img = Draw.DrawMorganBits(tpls, molsPerRow=5, legends=[str(x+1) for x in fp.GetOnBits()],
    #                                   drawOptions=drawOptions,
    #                                   subImgSize=(200, 200))
    #     # Draw.DrawMorganBit(Chem.AddHs(Chem.MolFromSmiles(smi)), [x for x in fp.GetOnBits()][1], bi, useSVG=True)
    #     img.save(str(j) + '.png')
    return des + list(GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))


global switch
switch = {'RDKit': rdkit, 'Atom Pairs': atom_pairs, 'Topological Torsions': topological_torsions,
          'MACCS keys': maccs_key, 'Morgan/Circular': morgan, '2D Pharmacophore': pharmacophore, 'Pattern': pattern,
          "one-hot": one_hot, "Dragon Descriptors": dragon_descriptors, '3D-Morse': Morse, 'WHIM': WHIM,
          'RDKit_WHIM': RDKit_WHIM, 'Morgan_WHIM': Morgan_WHIM, 'RDKit_Morse': RDKit_Morse,
          'Morgan_Morse': Morgan_Morse}


def generate_descriptors_file(fp):
    type_descriptors = fp
    map_pollutant_to_smiles = globalvaribale.get_value("MAP_POLLUTANT_TO_SMILES")
    # structural_descriptors_database = globalvaribale.get_value('structural_descriptors_path')
    smiles_list = [item[1] for item in map_pollutant_to_smiles.items()]
    value_list = []
    for smiles in smiles_list:
        value = switch.get(type_descriptors)(smiles)
        value_list.append(value)
    values = np.array(value_list)
    if type_descriptors == "Topological Torsions":
        title_ids = bag_of_topological_torsions()
    # elif type_descriptors == "Structural Descriptors":
    #     title_ids = structural_descriptors_database[0, 1:].tolist()
    else:
        title_ids = np.arange(0, values.shape[1], 1).tolist()
    title = copy.copy(title_ids)
    title.insert(0, "pollutant")
    values = np.c_[np.array(smiles_list)[:, np.newaxis], values]
    file = np.r_[np.array(title)[np.newaxis, :], values]
    np.save(os.getcwd() + "descriptors.npy", file)
    return file


def encoder(data, fp):
    file = generate_descriptors_file(fp)
    # print(file[:, 0])
    smiles = file[1:, 0]
    # print(smiles)
    descriptors = file[1:, 1:]
    descriptors = descriptors.astype(np.float)
    # print(descriptors.shape)
    # np.savetxt('des.txt', descriptors.T)
    mask = []
    # title = []
    for index in range(descriptors.shape[1]):
        if len(np.unique(descriptors[:, index])) == 1:
            mask.append(False)
        else:
            mask.append(True)
            # title.append(index+1)
    # print(mask)
    descriptors = descriptors[:, mask]
    # print(title)
    # print(descriptors.shape)
    smiles_to_descriptors = dict(zip(smiles, descriptors))
    pollutant_to_smiles = globalvaribale.get_value("MAP_POLLUTANT_TO_SMILES")
    func_smiles_to_descriptors = lambda x: smiles_to_descriptors[pollutant_to_smiles[x]]
    data_smiles = data[:, 0]
    data_descriptors = np.array(list(map(func_smiles_to_descriptors, data_smiles))).astype(np.float32)
    data = np.c_[data_descriptors, data[:, 1:]]
    return data


def random_label(data):
    number_sample = data.shape[0]
    mask = [index for index in range(number_sample)]
    np.random.shuffle(mask)
    data[:, -7:] = data[mask, -7:]
    return data


def train_test_split(data, ratio):
    number_sample = data.shape[0]
    number_test = int(number_sample*ratio)
    index_sample = [index for index in range(number_sample)]
    seed = globalvaribale.get_value("SEED")
    np.random.seed(seed)
    np.random.shuffle(index_sample)
    index_test = index_sample[0:number_test]
    index_train = index_sample[number_test:]
    return index_train, index_test


def list_reduce(list_1, list_2):
    list_result = []
    for element in list_1:
        if element not in list_2:
            list_result.append(element)
    return list_result


def k_fold_mask(number_sample, k_fold=5):
    mask = [index for index in range(number_sample)]
    seed = globalvaribale.get_value("SEED")
    np.random.seed(seed)
    np.random.shuffle(mask)
    mask_list = []
    gap = number_sample // k_fold
    for index in range(k_fold):
        mask_temp = mask[index*gap:(index+1)*gap]
        mask_list.append(mask_temp)
    return mask_list


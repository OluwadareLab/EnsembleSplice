import pandas as pd
from random import seed, shuffle
import copy

_seed = 123454

data_stats = {
    'Dataset Name: ':'',
    'Total number of positive acceptor sites: ':'',
    'Total number of false acceptor sites: ':'',
    'Total number of positive donor sites: ':'',
    'Total number of false donor sites: ':'',
    'Total pos:neg ratio for acceptor sites: ':'',
    'Total pos:neg ratio for donor sites: ':'',
    'Path to the positive acceptor file: ':'',
    'Path to the false acceptor file: ':'',
    'Path to the positive donor file: ':'',
    'Path to the false donor file: ':'',
    'Number of positive acceptor sites in the subset used for training: ':'',
    'Number of false acceptor sites in the subset used for training: ':'',
    'Number of positive donor sites in the subset used for training: ':'',
    'Number of false donor sites in the subset used for training: ':'',
    'pos:neg ratio for acceptor sites in the subset used for training: ':'',
    'pos:neg ratio for donor sites in the subset used for training: ':'',
    'Number of positive acceptor sites in the subset used for testing: ':'',
    'Number of false acceptor sites in the subset used for testing: ':'',
    'Number of positive donor sites in the subset used for testing: ':'',
    'Number of false donor sites in the subset used for testing: ':'',
    'pos:neg ratio for acceptor sites in the subset used for testing: ':'',
    'pos:neg ratio for donor sites in the subset used for testing: ':'',
    'The seed used to shuffle the data files before taking the subset: ':'',
}

def create_nn269():
    nn269_dict = copy.deepcopy(data_stats)
    # train
    trn_pos_acceptor_sites = './NN269/Train/Acceptor_Train_Positive.txt'
    trn_neg_acceptor_sites = './NN269/Train/Acceptor_Train_Negative.txt'
    trn_pos_donor_sites = './NN269/Train/Donor_Train_Positive.txt'
    trn_neg_donor_sites = './NN269/Train/Donor_Train_Negative.txt'
    # test
    tst_pos_acceptor_sites = './NN269/Test/Acceptor_Test_Positive.txt'
    tst_neg_acceptor_sites = './NN269/Test/Acceptor_Test_Negative.txt'
    tst_pos_donor_sites = './NN269/Test/Donor_Test_Positive.txt'
    tst_neg_donor_sites = './NN269/Test/Donor_Test_Negative.txt'

    nn269_dict['Dataset Name: '] = 'NN269'

    # train
    with open(trn_pos_acceptor_sites, 'r') as f:
        all_lines = f.readlines()
        trn_pos_acc_seq_data = [elt.replace('\n','') for elt in list(filter(lambda line: False if '>' in line else True, all_lines))]
        trn_pos_acc_len = len(trn_pos_acc_seq_data)
        nn269_dict['Number of positive acceptor sites in the subset used for training: '] = str(trn_pos_acc_len)
        f.close()

    with open(trn_neg_acceptor_sites, 'r') as f:
        all_lines = f.readlines()
        trn_neg_acc_seq_data = [elt.replace('\n','') for elt in list(filter(lambda line: False if '>' in line else True, all_lines))]
        trn_neg_acc_len = len(trn_neg_acc_seq_data)
        nn269_dict['Number of false acceptor sites in the subset used for training: '] = str(trn_neg_acc_len)
        f.close()

    nn269_dict['pos:neg ratio for acceptor sites in the subset used for training: '] = f'{trn_pos_acc_len/trn_pos_acc_len}:{trn_neg_acc_len/trn_pos_acc_len}'
    with open(trn_pos_donor_sites, 'r') as f:
        all_lines = f.readlines()
        trn_pos_don_seq_data = [elt.replace('\n','') for elt in list(filter(lambda line: False if '>' in line else True, all_lines))]
        trn_pos_don_len = len(trn_pos_don_seq_data)
        nn269_dict['Number of positive donor sites in the subset used for training: '] = str(trn_pos_don_len)
        f.close()

    with open(trn_neg_donor_sites, 'r') as f:
        all_lines = f.readlines()
        trn_neg_don_seq_data = [elt.replace('\n','') for elt in list(filter(lambda line: False if '>' in line else True, all_lines))]
        trn_neg_don_len = len(trn_neg_don_seq_data)
        nn269_dict['Number of false donor sites in the subset used for training: '] = str(trn_neg_don_len)
        f.close()
    nn269_dict['pos:neg ratio for donor sites in the subset used for training: '] = f'{trn_pos_don_len/trn_pos_don_len}:{trn_neg_don_len/trn_pos_don_len}'


    # test
    with open(tst_pos_acceptor_sites, 'r') as f:
        all_lines = f.readlines()
        tst_pos_acc_seq_data = [elt.replace('\n','') for elt in list(filter(lambda line: False if '>' in line else True, all_lines))]
        tst_pos_acc_len = len(tst_pos_acc_seq_data)
        nn269_dict['Number of positive acceptor sites in the subset used for testing: '] = str(tst_pos_acc_len)
        f.close()

    with open(tst_neg_acceptor_sites, 'r') as f:
        all_lines = f.readlines()
        tst_neg_acc_seq_data = [elt.replace('\n','') for elt in list(filter(lambda line: False if '>' in line else True, all_lines))]
        tst_neg_acc_len = len(tst_neg_acc_seq_data)
        nn269_dict['Number of false acceptor sites in the subset used for testing: '] = str(tst_neg_acc_len)
        f.close()

    nn269_dict['pos:neg ratio for acceptor sites in the subset used for testing: '] = f'{tst_pos_acc_len/tst_pos_acc_len}:{tst_neg_acc_len/tst_pos_acc_len}'
    with open(tst_pos_donor_sites, 'r') as f:
        all_lines = f.readlines()
        tst_pos_don_seq_data = [elt.replace('\n','') for elt in list(filter(lambda line: False if '>' in line else True, all_lines))]
        tst_pos_don_len = len(tst_pos_don_seq_data)
        nn269_dict['Number of positive donor sites in the subset used for testing: '] = str(tst_pos_don_len)
        f.close()

    with open(tst_neg_donor_sites, 'r') as f:
        all_lines = f.readlines()
        tst_neg_don_seq_data = [elt.replace('\n','') for elt in list(filter(lambda line: False if '>' in line else True, all_lines))]
        tst_neg_don_len = len(tst_neg_don_seq_data)
        nn269_dict['Number of false donor sites in the subset used for testing: '] = str(tst_neg_don_len)
        f.close()
    nn269_dict['pos:neg ratio for donor sites in the subset used for testing: '] = f'{tst_pos_don_len/tst_pos_don_len}:{tst_neg_don_len/tst_pos_don_len}'

    # overall
    nn269_dict['Total pos:neg ratio for acceptor sites: '] = f'{(trn_pos_acc_len+tst_pos_acc_len)/(trn_pos_acc_len+tst_pos_acc_len)}:{(trn_neg_acc_len+tst_neg_acc_len)/(trn_pos_acc_len+tst_pos_acc_len)}'
    nn269_dict['Total pos:neg ratio for donor sites: '] = f'{(trn_pos_don_len+tst_pos_don_len)/(trn_pos_don_len+tst_pos_don_len)}:{(trn_neg_don_len+tst_neg_don_len)/(trn_pos_don_len+tst_pos_don_len)}'
    nn269_dict['Total number of positive acceptor sites: '] = f'{trn_pos_acc_len+tst_pos_acc_len}'
    nn269_dict['Total number of positive donor sites: '] = f'{trn_pos_don_len+tst_pos_don_len}'
    nn269_dict['Total number of false acceptor sites: '] = f'{trn_neg_acc_len+tst_neg_acc_len}'
    nn269_dict['Total number of false donor sites: '] = f'{trn_neg_don_len+tst_neg_don_len}'
    nn269_dict['Path to the positive acceptor file: '] = f'{trn_pos_acceptor_sites},{tst_pos_acceptor_sites}'
    nn269_dict['Path to the false acceptor file: '] = f'{trn_neg_acceptor_sites},{tst_neg_acceptor_sites}'
    nn269_dict['Path to the positive donor file: '] = f'{trn_pos_donor_sites},{tst_pos_donor_sites}'
    nn269_dict['Path to the false donor file: '] = f'{trn_neg_donor_sites},{tst_neg_donor_sites}'

    nn269_dict['The seed used to shuffle the data files before taking the subset: '] = 'None'

    with open('dataset_stats.txt', 'a') as f:
        for key, value in nn269_dict.items():
            f.write(f'{key}{value}')
            f.write('\n')
        f.close()


def create_hs3d_subset():

    hs3d_dict = copy.deepcopy(data_stats)

    # original
    org_pos_acceptor_sites = './HS3D/Original/Acceptor_Positive.txt'
    org_neg_acceptor_sites = './HS3D/Original/Acceptor_Negative03.txt'
    org_pos_donor_sites = './HS3D/Original/Donor_Positive.txt'
    org_neg_donor_sites = './HS3D/Original/Donor_Negative02.txt'

    # train
    trn_pos_acceptor_sites = './HS3D/SubsetTrain/Acceptor_Train_Positive.txt'
    trn_neg_acceptor_sites = './HS3D/SubsetTrain/Acceptor_Train_Negative.txt'
    trn_pos_donor_sites = './HS3D/SubsetTrain/Donor_Train_Positive.txt'
    trn_neg_donor_sites = './HS3D/SubsetTrain/Donor_Train_Negative.txt'

    # test
    tst_pos_acceptor_sites = './HS3D/SubsetTest/Acceptor_Test_Positive.txt'
    tst_neg_acceptor_sites = './HS3D/SubsetTest/Acceptor_Test_Negative.txt'
    tst_pos_donor_sites = './HS3D/SubsetTest/Donor_Test_Positive.txt'
    tst_neg_donor_sites = './HS3D/SubsetTest/Donor_Test_Negative.txt'

    hs3d_dict['Dataset Name: '] = 'HS3D'
    hs3d_dict['The seed used to shuffle the data files before taking the subset: '] = str(_seed)
    hs3d_dict['Path to the positive acceptor file: '] = f'[{org_pos_acceptor_sites}, {trn_pos_acceptor_sites}, {tst_pos_acceptor_sites}]'
    hs3d_dict['Path to the false acceptor file: '] = f'[{org_neg_acceptor_sites }, {trn_neg_acceptor_sites}, {tst_pos_acceptor_sites}]'
    hs3d_dict['Path to the positive donor file: '] = f'[{org_pos_donor_sites}, {trn_pos_donor_sites}, {tst_pos_donor_sites}]'
    hs3d_dict['Path to the false donor file: '] = f'[{org_neg_donor_sites}, {trn_neg_donor_sites}, {tst_neg_donor_sites}]'

    # original
    with open(org_pos_acceptor_sites, 'r') as f:
        all_lines = f.readlines()[4:]
        pos_acc_seq_data = [elt.split(':')[1].replace('\n','').replace(' ','') for elt in all_lines]
        pos_acc_len = len(pos_acc_seq_data)
        hs3d_dict['Total number of positive acceptor sites: '] = str(pos_acc_len)
        seed(_seed)
        shuffle(pos_acc_seq_data)
        trn_pos_acc_seq_data = pos_acc_seq_data[:round(pos_acc_len*0.80)]
        tst_pos_acc_seq_data = pos_acc_seq_data[round(pos_acc_len*0.80):]
        trn_pos_acc_len = len(trn_pos_acc_seq_data)
        tst_pos_acc_len = len(tst_pos_acc_seq_data)
        hs3d_dict['Number of positive acceptor sites in the subset used for training: '] = str(trn_pos_acc_len)
        hs3d_dict['Number of positive acceptor sites in the subset used for testing: '] = str(tst_pos_acc_len)
        f.close()

    with open(org_neg_acceptor_sites, 'r') as f:
        all_lines = f.readlines()[1:-1]
        neg_acc_seq_data = [elt.split(':')[1].replace('\n','').replace(' ','') for elt in all_lines]
        neg_acc_len = len(neg_acc_seq_data)
        hs3d_dict['Total number of false acceptor sites: '] = str(neg_acc_len)
        seed(_seed)
        shuffle(neg_acc_seq_data)
        neg_acc_subset = neg_acc_seq_data[:10000]
        trn_neg_acc_seq_data = neg_acc_subset[:round(len(neg_acc_subset)*0.80)]
        tst_neg_acc_seq_data = neg_acc_subset[round(len(neg_acc_subset)*0.80):]
        trn_neg_acc_len = len(trn_neg_acc_seq_data)
        tst_neg_acc_len = len(tst_neg_acc_seq_data)
        hs3d_dict['Number of false acceptor sites in the subset used for training: '] = str(trn_neg_acc_len)
        hs3d_dict['Number of false acceptor sites in the subset used for testing: '] = str(tst_neg_acc_len)
        f.close()

    hs3d_dict['Total pos:neg ratio for acceptor sites: '] = f'{pos_acc_len/pos_acc_len}:{neg_acc_len/pos_acc_len}'
    hs3d_dict['pos:neg ratio for acceptor sites in the subset used for training: '] = f'{trn_pos_acc_len/trn_pos_acc_len}:{trn_neg_acc_len/trn_pos_acc_len}'
    hs3d_dict['pos:neg ratio for acceptor sites in the subset used for testing: '] = f'{tst_pos_acc_len/tst_pos_acc_len}:{tst_neg_acc_len/tst_pos_acc_len}'

    with open(org_pos_donor_sites, 'r') as f:
        all_lines = f.readlines()[4:]
        pos_don_seq_data = [elt.split(':')[1].replace('\n','').replace(' ','') for elt in all_lines]
        pos_don_len = len(pos_don_seq_data)
        hs3d_dict['Total number of positive donor sites: '] = str(pos_don_len)
        seed(_seed)
        shuffle(pos_don_seq_data)
        trn_pos_don_seq_data = pos_don_seq_data[:round(pos_don_len*0.80)]
        tst_pos_don_seq_data = pos_don_seq_data[round(pos_don_len*0.80):]
        trn_pos_don_len = len(trn_pos_don_seq_data)
        tst_pos_don_len = len(tst_pos_don_seq_data)
        hs3d_dict['Number of positive donor sites in the subset used for training: '] = str(trn_pos_don_len)
        hs3d_dict['Number of positive donor sites in the subset used for testing: '] = str(tst_pos_don_len)
        f.close()

    with open(org_neg_donor_sites, 'r') as f:
        all_lines = f.readlines()[1:-1]
        neg_don_seq_data = [elt.split(':')[1].replace('\n','').replace(' ','') for elt in all_lines]
        neg_don_len = len(neg_don_seq_data)
        hs3d_dict['Total number of false donor sites: '] = str(neg_don_len)
        seed(_seed)
        shuffle(neg_don_seq_data)
        neg_don_subset = neg_don_seq_data[:10000]
        trn_neg_don_seq_data = neg_don_subset[:round(len(neg_don_subset)*0.80)]
        tst_neg_don_seq_data = neg_don_subset[round(len(neg_don_subset)*0.80):]
        trn_neg_don_len = len(trn_neg_don_seq_data)
        tst_neg_don_len = len(tst_neg_don_seq_data)
        hs3d_dict['Number of false donor sites in the subset used for training: '] = str(trn_neg_don_len)
        hs3d_dict['Number of false donor sites in the subset used for testing: '] = str(tst_neg_don_len)
        f.close()

    hs3d_dict['Total pos:neg ratio for donor sites: '] = f'{pos_don_len/pos_don_len}:{neg_don_len/pos_don_len}'
    hs3d_dict['pos:neg ratio for donor sites in the subset used for training: '] = f'{trn_pos_don_len/trn_pos_don_len}:{trn_neg_don_len/trn_pos_don_len}'
    hs3d_dict['pos:neg ratio for donor sites in the subset used for testing: '] = f'{tst_pos_don_len/tst_pos_don_len}:{tst_neg_don_len/tst_pos_don_len}'

    # train
    with open(trn_pos_acceptor_sites, 'w') as f:
        for index, line in enumerate(trn_pos_acc_seq_data):
            f.write(f'> seq {index+1}')
            f.write('\n')
            f.write(line)
            f.write('\n')
        f.close()

    with open(trn_neg_acceptor_sites, 'w') as f:
        for index, line in enumerate(trn_neg_acc_seq_data):
            f.write(f'> seq {index+1}')
            f.write('\n')
            f.write(line)
            f.write('\n')
        f.close()

    with open(trn_pos_donor_sites, 'w') as f:
        for index, line in enumerate(trn_pos_don_seq_data):
            f.write(f'> seq {index+1}')
            f.write('\n')
            f.write(line)
            f.write('\n')
        f.close()

    with open(trn_neg_donor_sites, 'w') as f:
        for index, line in enumerate(trn_neg_don_seq_data):
            f.write(f'> seq {index+1}')
            f.write('\n')
            f.write(line)
            f.write('\n')
        f.close()

    # test
    with open(tst_pos_acceptor_sites, 'w') as f:
        for index, line in enumerate(tst_pos_acc_seq_data):
            f.write(f'> seq {index+1}')
            f.write('\n')
            f.write(line)
            f.write('\n')
        f.close()

    with open(tst_neg_acceptor_sites, 'w') as f:
        for index, line in enumerate(tst_neg_acc_seq_data):
            f.write(f'> seq {index+1}')
            f.write('\n')
            f.write(line)
            f.write('\n')
        f.close()

    with open(tst_pos_donor_sites, 'w') as f:
        for index, line in enumerate(tst_pos_don_seq_data):
            f.write(f'> seq {index+1}')
            f.write('\n')
            f.write(line)
            f.write('\n')
        f.close()

    with open(tst_neg_donor_sites, 'w') as f:
        for index, line in enumerate(tst_neg_don_seq_data):
            f.write(f'> seq {index+1}')
            f.write('\n')
            f.write(line)
            f.write('\n')
        f.close()


    with open('dataset_stats.txt', 'a') as f:
        for key, value in hs3d_dict.items():
            f.write(f'{key}{value}')
            f.write('\n')
        f.close()

def create_homo_subset():
    homo_dict = copy.deepcopy(data_stats)

    # original
    org_pos_acceptor_sites = './Homo_sapiens/Original/positive_DNA_seqs_acceptor_hs.fa'
    org_neg_acceptor_sites =  './Homo_sapiens/Original/negative_DNA_seqs_acceptor_hs.fa'
    org_pos_donor_sites = './Homo_sapiens/Original/positive_DNA_seqs_donor_hs.fa'
    org_neg_donor_sites = './Homo_sapiens/Original/negative_DNA_seqs_donor_hs.fa'

    # train
    trn_pos_acceptor_sites = './Homo_sapiens/SubsetTrain/Acceptor_Train_Positive.txt'
    trn_neg_acceptor_sites = './Homo_sapiens/SubsetTrain/Acceptor_Train_Negative.txt'
    trn_pos_donor_sites = './Homo_sapiens/SubsetTrain/Donor_Train_Positive.txt'
    trn_neg_donor_sites = './Homo_sapiens/SubsetTrain/Donor_Train_Negative.txt'

    # test
    tst_pos_acceptor_sites = './Homo_sapiens/SubsetTest/Acceptor_Test_Positive.txt'
    tst_neg_acceptor_sites = './Homo_sapiens/SubsetTest/Acceptor_Test_Negative.txt'
    tst_pos_donor_sites = './Homo_sapiens/SubsetTest/Donor_Test_Positive.txt'
    tst_neg_donor_sites = './Homo_sapiens/SubsetTest/Donor_Test_Negative.txt'

    homo_dict['Dataset Name: '] = 'Homo sapiens'
    homo_dict['The seed used to shuffle the data files before taking the subset: '] = str(_seed)
    homo_dict['Path to the positive acceptor file: '] = f'[{org_pos_acceptor_sites}, {trn_pos_acceptor_sites}, {tst_pos_acceptor_sites}]'
    homo_dict['Path to the false acceptor file: '] = f'[{org_neg_acceptor_sites }, {trn_neg_acceptor_sites}, {tst_pos_acceptor_sites}]'
    homo_dict['Path to the positive donor file: '] = f'[{org_pos_donor_sites}, {trn_pos_donor_sites}, {tst_pos_donor_sites}]'
    homo_dict['Path to the false donor file: '] = f'[{org_neg_donor_sites}, {trn_neg_donor_sites}, {tst_neg_donor_sites}]'

    # original
    with open(org_pos_acceptor_sites, 'r') as f:
        all_lines = f.readlines()
        pos_acc_seq_data = [elt.replace('\n','').replace(' ','') for elt in all_lines]
        pos_acc_len = len(pos_acc_seq_data)
        homo_dict['Total number of positive acceptor sites: '] = str(pos_acc_len)
        seed(_seed)
        shuffle(pos_acc_seq_data)
        pos_acc_subset = pos_acc_seq_data[:8000]
        trn_pos_acc_seq_data = pos_acc_subset[:round(len(pos_acc_subset)*0.80)]
        tst_pos_acc_seq_data = pos_acc_subset[round(len(pos_acc_subset)*0.80):]
        trn_pos_acc_len = len(trn_pos_acc_seq_data)
        tst_pos_acc_len = len(tst_pos_acc_seq_data)
        homo_dict['Number of positive acceptor sites in the subset used for training: '] = str(trn_pos_acc_len)
        homo_dict['Number of positive acceptor sites in the subset used for testing: '] = str(tst_pos_acc_len)
        f.close()

    with open(org_neg_acceptor_sites, 'r') as f:
        all_lines = f.readlines()
        neg_acc_seq_data = [elt.replace('\n','').replace(' ','') for elt in all_lines]
        neg_acc_len = len(neg_acc_seq_data)
        homo_dict['Total number of false acceptor sites: '] = str(neg_acc_len)
        seed(_seed)
        shuffle(neg_acc_seq_data)
        neg_acc_subset = neg_acc_seq_data[:8000]
        trn_neg_acc_seq_data = neg_acc_subset[:round(len(neg_acc_subset)*0.80)]
        tst_neg_acc_seq_data = neg_acc_subset[round(len(neg_acc_subset)*0.80):]
        trn_neg_acc_len = len(trn_neg_acc_seq_data)
        tst_neg_acc_len = len(tst_neg_acc_seq_data)
        homo_dict['Number of false acceptor sites in the subset used for training: '] = str(trn_neg_acc_len)
        homo_dict['Number of false acceptor sites in the subset used for testing: '] = str(tst_neg_acc_len)
        f.close()

    homo_dict['Total pos:neg ratio for acceptor sites: '] = f'{pos_acc_len/pos_acc_len}:{neg_acc_len/pos_acc_len}'
    homo_dict['pos:neg ratio for acceptor sites in the subset used for training: '] = f'{trn_pos_acc_len/trn_pos_acc_len}:{trn_neg_acc_len/trn_pos_acc_len}'
    homo_dict['pos:neg ratio for acceptor sites in the subset used for testing: '] = f'{tst_pos_acc_len/tst_pos_acc_len}:{tst_neg_acc_len/tst_pos_acc_len}'

    with open(org_pos_donor_sites, 'r') as f:
        all_lines = f.readlines()
        pos_don_seq_data = [elt.replace('\n','').replace(' ','') for elt in all_lines]
        pos_don_len = len(pos_don_seq_data)
        homo_dict['Total number of positive donor sites: '] = str(pos_don_len)
        seed(_seed)
        shuffle(pos_don_seq_data)
        pos_don_subset = pos_don_seq_data[:8000]
        trn_pos_don_seq_data = pos_don_subset[:round(len(pos_don_subset)*0.80)]
        tst_pos_don_seq_data = pos_don_subset[round(len(pos_don_subset)*0.80):]
        trn_pos_don_len = len(trn_pos_don_seq_data)
        tst_pos_don_len = len(tst_pos_don_seq_data)
        homo_dict['Number of positive donor sites in the subset used for training: '] = str(trn_pos_don_len)
        homo_dict['Number of positive donor sites in the subset used for testing: '] = str(tst_pos_don_len)
        f.close()

    with open(org_neg_donor_sites, 'r') as f:
        all_lines = f.readlines()
        neg_don_seq_data = [elt.replace('\n','').replace(' ','') for elt in all_lines]
        neg_don_len = len(neg_don_seq_data)
        homo_dict['Total number of false donor sites: '] = str(neg_don_len)
        seed(_seed)
        shuffle(neg_don_seq_data)
        neg_don_subset = neg_don_seq_data[:8000]
        trn_neg_don_seq_data = neg_don_subset[:round(len(neg_don_subset)*0.80)]
        tst_neg_don_seq_data = neg_don_subset[round(len(neg_don_subset)*0.80):]
        trn_neg_don_len = len(trn_neg_don_seq_data)
        tst_neg_don_len = len(tst_neg_don_seq_data)
        homo_dict['Number of false donor sites in the subset used for training: '] = str(trn_neg_don_len)
        homo_dict['Number of false donor sites in the subset used for testing: '] = str(tst_neg_don_len)
        f.close()

    homo_dict['Total pos:neg ratio for donor sites: '] = f'{pos_don_len/pos_don_len}:{neg_don_len/pos_don_len}'
    homo_dict['pos:neg ratio for donor sites in the subset used for training: '] = f'{trn_pos_don_len/trn_pos_don_len}:{trn_neg_don_len/trn_pos_don_len}'
    homo_dict['pos:neg ratio for donor sites in the subset used for testing: '] = f'{tst_pos_don_len/tst_pos_don_len}:{tst_neg_don_len/tst_pos_don_len}'

    # train
    with open(trn_pos_acceptor_sites, 'w') as f:
        for index, line in enumerate(trn_pos_acc_seq_data):
            f.write(f'> seq {index+1}')
            f.write('\n')
            f.write(line)
            f.write('\n')
        f.close()

    with open(trn_neg_acceptor_sites, 'w') as f:
        for index, line in enumerate(trn_neg_acc_seq_data):
            f.write(f'> seq {index+1}')
            f.write('\n')
            f.write(line)
            f.write('\n')
        f.close()

    with open(trn_pos_donor_sites, 'w') as f:
        for index, line in enumerate(trn_pos_don_seq_data):
            f.write(f'> seq {index+1}')
            f.write('\n')
            f.write(line)
            f.write('\n')
        f.close()

    with open(trn_neg_donor_sites, 'w') as f:
        for index, line in enumerate(trn_neg_don_seq_data):
            f.write(f'> seq {index+1}')
            f.write('\n')
            f.write(line)
            f.write('\n')
        f.close()

    # test
    with open(tst_pos_acceptor_sites, 'w') as f:
        for index, line in enumerate(tst_pos_acc_seq_data):
            f.write(f'> seq {index+1}')
            f.write('\n')
            f.write(line)
            f.write('\n')
        f.close()

    with open(tst_neg_acceptor_sites, 'w') as f:
        for index, line in enumerate(tst_neg_acc_seq_data):
            f.write(f'> seq {index+1}')
            f.write('\n')
            f.write(line)
            f.write('\n')
        f.close()

    with open(tst_pos_donor_sites, 'w') as f:
        for index, line in enumerate(tst_pos_don_seq_data):
            f.write(f'> seq {index+1}')
            f.write('\n')
            f.write(line)
            f.write('\n')
        f.close()

    with open(tst_neg_donor_sites, 'w') as f:
        for index, line in enumerate(tst_neg_don_seq_data):
            f.write(f'> seq {index+1}')
            f.write('\n')
            f.write(line)
            f.write('\n')
        f.close()


    with open('dataset_stats.txt', 'a') as f:
        for key, value in homo_dict.items():
            f.write(f'{key}{value}')
            f.write('\n')
        f.close()

def create_arab_subset():

    arab_dict = copy.deepcopy(data_stats)

    # original
    org_pos_acceptor_sites = './Arabidopsis/Original/positive_DNA_seqs_acceptor_at.fa'
    org_neg_acceptor_sites =  './Arabidopsis/Original/negative_DNA_seqs_acceptor_at.fa'
    org_pos_donor_sites = './Arabidopsis/Original/positive_DNA_seqs_donor_at.fa'
    org_neg_donor_sites = './Arabidopsis/Original/negative_DNA_seqs_donor_at.fa'

    # train
    trn_pos_acceptor_sites = './Arabidopsis/SubsetTrain/Acceptor_Train_Positive.txt'
    trn_neg_acceptor_sites = './Arabidopsis/SubsetTrain/Acceptor_Train_Negative.txt'
    trn_pos_donor_sites = './Arabidopsis/SubsetTrain/Donor_Train_Positive.txt'
    trn_neg_donor_sites = './Arabidopsis/SubsetTrain/Donor_Train_Negative.txt'

    # test
    tst_pos_acceptor_sites = './Arabidopsis/SubsetTest/Acceptor_Test_Positive.txt'
    tst_neg_acceptor_sites = './Arabidopsis/SubsetTest/Acceptor_Test_Negative.txt'
    tst_pos_donor_sites = './Arabidopsis/SubsetTest/Donor_Test_Positive.txt'
    tst_neg_donor_sites = './Arabidopsis/SubsetTest/Donor_Test_Negative.txt'

    arab_dict['Dataset Name: '] = 'Arabidopsis thaliana'
    arab_dict['The seed used to shuffle the data files before taking the subset: '] = str(_seed)
    arab_dict['Path to the positive acceptor file: '] = f'[{org_pos_acceptor_sites}, {trn_pos_acceptor_sites}, {tst_pos_acceptor_sites}]'
    arab_dict['Path to the false acceptor file: '] = f'[{org_neg_acceptor_sites }, {trn_neg_acceptor_sites}, {tst_pos_acceptor_sites}]'
    arab_dict['Path to the positive donor file: '] = f'[{org_pos_donor_sites}, {trn_pos_donor_sites}, {tst_pos_donor_sites}]'
    arab_dict['Path to the false donor file: '] = f'[{org_neg_donor_sites}, {trn_neg_donor_sites}, {tst_neg_donor_sites}]'

    # original
    with open(org_pos_acceptor_sites, 'r') as f:
        all_lines = f.readlines()
        pos_acc_seq_data = [elt.replace('\n','').replace(' ','') for elt in all_lines]
        pos_acc_len = len(pos_acc_seq_data)
        arab_dict['Total number of positive acceptor sites: '] = str(pos_acc_len)
        seed(_seed)
        shuffle(pos_acc_seq_data)
        pos_acc_subset = pos_acc_seq_data[:8000]
        trn_pos_acc_seq_data = pos_acc_subset[:round(len(pos_acc_subset)*0.80)]
        tst_pos_acc_seq_data = pos_acc_subset[round(len(pos_acc_subset)*0.80):]
        trn_pos_acc_len = len(trn_pos_acc_seq_data)
        tst_pos_acc_len = len(tst_pos_acc_seq_data)
        arab_dict['Number of positive acceptor sites in the subset used for training: '] = str(trn_pos_acc_len)
        arab_dict['Number of positive acceptor sites in the subset used for testing: '] = str(tst_pos_acc_len)
        f.close()

    with open(org_neg_acceptor_sites, 'r') as f:
        all_lines = f.readlines()
        neg_acc_seq_data = [elt.replace('\n','').replace(' ','') for elt in all_lines]
        neg_acc_len = len(neg_acc_seq_data)
        arab_dict['Total number of false acceptor sites: '] = str(neg_acc_len)
        seed(_seed)
        shuffle(neg_acc_seq_data)
        neg_acc_subset = neg_acc_seq_data[:8000]
        trn_neg_acc_seq_data = neg_acc_subset[:round(len(neg_acc_subset)*0.80)]
        tst_neg_acc_seq_data = neg_acc_subset[round(len(neg_acc_subset)*0.80):]
        trn_neg_acc_len = len(trn_neg_acc_seq_data)
        tst_neg_acc_len = len(tst_neg_acc_seq_data)
        arab_dict['Number of false acceptor sites in the subset used for training: '] = str(trn_neg_acc_len)
        arab_dict['Number of false acceptor sites in the subset used for testing: '] = str(tst_neg_acc_len)
        f.close()

    arab_dict['Total pos:neg ratio for acceptor sites: '] = f'{pos_acc_len/pos_acc_len}:{neg_acc_len/pos_acc_len}'
    arab_dict['pos:neg ratio for acceptor sites in the subset used for training: '] = f'{trn_pos_acc_len/trn_pos_acc_len}:{trn_neg_acc_len/trn_pos_acc_len}'
    arab_dict['pos:neg ratio for acceptor sites in the subset used for testing: '] = f'{tst_pos_acc_len/tst_pos_acc_len}:{tst_neg_acc_len/tst_pos_acc_len}'

    with open(org_pos_donor_sites, 'r') as f:
        all_lines = f.readlines()
        pos_don_seq_data = [elt.replace('\n','').replace(' ','') for elt in all_lines]
        pos_don_len = len(pos_don_seq_data)
        arab_dict['Total number of positive donor sites: '] = str(pos_don_len)
        seed(_seed)
        shuffle(pos_don_seq_data)
        pos_don_subset = pos_don_seq_data[:8000]
        trn_pos_don_seq_data = pos_don_subset[:round(len(pos_don_subset)*0.80)]
        tst_pos_don_seq_data = pos_don_subset[round(len(pos_don_subset)*0.80):]
        trn_pos_don_len = len(trn_pos_don_seq_data)
        tst_pos_don_len = len(tst_pos_don_seq_data)
        arab_dict['Number of positive donor sites in the subset used for training: '] = str(trn_pos_don_len)
        arab_dict['Number of positive donor sites in the subset used for testing: '] = str(tst_pos_don_len)
        f.close()

    with open(org_neg_donor_sites, 'r') as f:
        all_lines = f.readlines()
        neg_don_seq_data = [elt.replace('\n','').replace(' ','') for elt in all_lines]
        neg_don_len = len(neg_don_seq_data)
        arab_dict['Total number of false donor sites: '] = str(neg_don_len)
        seed(_seed)
        shuffle(neg_don_seq_data)
        neg_don_subset = neg_don_seq_data[:8000]
        trn_neg_don_seq_data = neg_don_subset[:round(len(neg_don_subset)*0.80)]
        tst_neg_don_seq_data = neg_don_subset[round(len(neg_don_subset)*0.80):]
        trn_neg_don_len = len(trn_neg_don_seq_data)
        tst_neg_don_len = len(tst_neg_don_seq_data)
        arab_dict['Number of false donor sites in the subset used for training: '] = str(trn_neg_don_len)
        arab_dict['Number of false donor sites in the subset used for testing: '] = str(tst_neg_don_len)
        f.close()

    arab_dict['Total pos:neg ratio for donor sites: '] = f'{pos_don_len/pos_don_len}:{neg_don_len/pos_don_len}'
    arab_dict['pos:neg ratio for donor sites in the subset used for training: '] = f'{trn_pos_don_len/trn_pos_don_len}:{trn_neg_don_len/trn_pos_don_len}'
    arab_dict['pos:neg ratio for donor sites in the subset used for testing: '] = f'{tst_pos_don_len/tst_pos_don_len}:{tst_neg_don_len/tst_pos_don_len}'

    # train
    with open(trn_pos_acceptor_sites, 'w') as f:
        for index, line in enumerate(trn_pos_acc_seq_data):
            f.write(f'> seq {index+1}')
            f.write('\n')
            f.write(line)
            f.write('\n')
        f.close()

    with open(trn_neg_acceptor_sites, 'w') as f:
        for index, line in enumerate(trn_neg_acc_seq_data):
            f.write(f'> seq {index+1}')
            f.write('\n')
            f.write(line)
            f.write('\n')
        f.close()

    with open(trn_pos_donor_sites, 'w') as f:
        for index, line in enumerate(trn_pos_don_seq_data):
            f.write(f'> seq {index+1}')
            f.write('\n')
            f.write(line)
            f.write('\n')
        f.close()

    with open(trn_neg_donor_sites, 'w') as f:
        for index, line in enumerate(trn_neg_don_seq_data):
            f.write(f'> seq {index+1}')
            f.write('\n')
            f.write(line)
            f.write('\n')
        f.close()

    # test
    with open(tst_pos_acceptor_sites, 'w') as f:
        for index, line in enumerate(tst_pos_acc_seq_data):
            f.write(f'> seq {index+1}')
            f.write('\n')
            f.write(line)
            f.write('\n')
        f.close()

    with open(tst_neg_acceptor_sites, 'w') as f:
        for index, line in enumerate(tst_neg_acc_seq_data):
            f.write(f'> seq {index+1}')
            f.write('\n')
            f.write(line)
            f.write('\n')
        f.close()

    with open(tst_pos_donor_sites, 'w') as f:
        for index, line in enumerate(tst_pos_don_seq_data):
            f.write(f'> seq {index+1}')
            f.write('\n')
            f.write(line)
            f.write('\n')
        f.close()

    with open(tst_neg_donor_sites, 'w') as f:
        for index, line in enumerate(tst_neg_don_seq_data):
            f.write(f'> seq {index+1}')
            f.write('\n')
            f.write(line)
            f.write('\n')
        f.close()


    with open('dataset_stats.txt', 'a') as f:
        for key, value in arab_dict.items():
            f.write(f'{key}{value}')
            f.write('\n')
        f.close()

def create_dros_subset():
    pass

def create_ce_subset():
    pass

def main():

    

    # if these are commented out, that means that they have been run already
    # create_nn269()
    # create_hs3d_subset()
    # create_homo_subset()
    # create_arab_subset()


    #seed(1232423423)
    # HOMO
    # targets = [
    #     '../../Homo/Original/negative_DNA_seqs_donor_hs.fa',
    #     '../../Homo/Original/positive_DNA_seqs_donor_hs.fa',
    #     '../../Homo/Original/negative_DNA_seqs_acceptor_hs.fa',
    #     '../../Homo/Original/positive_DNA_seqs_acceptor_hs.fa',
    # ]
    # train_dests = [
    #     '../../Homo/Train/neg_donor_hs_train.fa',
    #     '../../Homo/Train/pos_donor_hs_train.fa',
    #     '../../Homo/Train/neg_acceptor_hs_train.fa',
    #     '../../Homo/Train/pos_acceptor_hs_train.fa',
    # ]
    # test_dests = [
    #     '../../Homo/Test/neg_donor_hs_test.fa',
    #     '../../Homo/Test/pos_donor_hs_test.fa',
    #     '../../Homo/Test/neg_acceptor_hs_test.fa',
    #     '../../Homo/Test/pos_acceptor_hs_test.fa',
    # ]
    #
    # for index, target in enumerate(targets):
    #     with open(target, 'r') as f:
    #         lines = f.readlines()
    #         shuffle(lines)
    #         f.close()
    #     train = lines[:round(len(lines)*0.8)]
    #     test = lines[round(len(lines)*0.8):]
    #     with open(train_dests[index], 'w') as f:
    #         f.writelines(train)
    #         f.close()
    #     with open(test_dests[index], 'w') as f:
    #         f.writelines(test)
    #         f.close()

    # HS3D

  # targets = [
  #     './HS3D/Original/Acceptor_Negative01.txt',
  #     './HS3D/Original/Donor_Negative01.txt',
  #     './HS3D/Original/Acceptor_Positive.txt',
  #     './HS3D/Original/Donor_Positive.txt',
  # ]
  # train_dests = [
  #     './HS3D/SubsetTrain/Acceptor_Train_Negative01.txt',
  #     './HS3D/SubsetTrain/Donor_Train_Negative01.txt',
  #     './HS3D/SubsetTrain/Acceptor_Train_Positive.txt',
  #     './HS3D/SubsetTrain/Donor_Train_Positive.txt',
  # ]
  # test_dests = [
  #     './HS3D/SubsetTest/Acceptor_Test_Negative01.txt',
  #     './HS3D/SubsetTest/Donor_Test_Negative01.txt',
  #     './HS3D/SubsetTest/Acceptor_Test_Positive.txt',
  #    './HS3D/SubsetTest/Donor_Test_Positive.txt',
  # ]
  # read in first acceptor pos, neg and first donor pos, neg
  # take 10000 instances random sampled
  # for index, target in enumerate(targets):
  # with open('./HS3D/SubsetTest/Acceptor_Test_Positive.txt', 'r') as f:
  #     lines = f.readlines()[4:]
  #     lines = lines[:-1] # remove the last incomplete line
  #     lines = [elt.split(':')[1].replace('\n','').replace(' ','') for elt in lines]
  #     seed(123432)
  #     shuffle(lines)
  #     f.close()
  # if 'Neg' in target:
  #     train = lines[:10000]
  #     test = lines[10000:12000]
  # else:
      # train = lines[:round(len(lines)*0.8)]
      # test = lines[round(len(lines)*0.8):]
  # with open('issCNN_acceptor_positive_test.fa', 'w') as f:
  #     for index, x in enumerate(lines):
  #         f.write(f'>seq{index+1}\n{x}\n')
  #     f.close()
      #assert len([elt for elt in train if '(' not in elt])==0,f'fuck'
      # with open(train_dests[index], 'w') as f:
      #     f.writelines(train)
      #     f.close()
      # with open(test_dests[index], 'w') as f:
      #     f.writelines(test)
      #     f.close()





    # # CE
    # targets = [
    #     './CE/Original/Acceptor_All.txt',
    #     './CE/Original/Donor_All.txt',
    # ]
    # train_dests = [
    #     './CE/Train/Acceptor_Train.txt',
    #     './CE/Train/Donor_Train.txt',
    # ]
    # test_dests = [
    #     './CE/Test/Acceptor_Test.txt',
    #     './CE/Test/Donor_Test.txt',
    # ]
    #
    # for index, target in enumerate(targets):
    #     with open(target, 'r') as f:
    #         lines = f.readlines()
    #         shuffle(lines)
    #         f.close()
    #     train = lines[:round(len(lines)*0.8)]
    #     test = lines[round(len(lines)*0.8):]
    #     with open(train_dests[index], 'w') as f:
    #         f.writelines(train)
    #         f.close()
    #     with open(test_dests[index], 'w') as f:
    #         f.writelines(test)
    #         f.close()

    # # HOMO SUBSET
    # targets = [
    #     '../../StorageEnsembleSplice/LargerDatasets/Homo/Original/negative_DNA_seqs_donor_hs.fa',
    #     '../../StorageEnsembleSplice/LargerDatasets/Homo/Original/positive_DNA_seqs_donor_hs.fa',
    #     '../../StorageEnsembleSplice/LargerDatasets/Homo/Original/negative_DNA_seqs_acceptor_hs.fa',
    #     '../../StorageEnsembleSplice/LargerDatasets/Homo/Original/positive_DNA_seqs_acceptor_hs.fa',
    # ]
    # train_dests = [
    #     './SubsetHomo/SubsetTrain/neg_donor_homo_train.txt',
    #     './SubsetHomo/SubsetTrain/pos_donor_homo_train.txt',
    #     './SubsetHomo/SubsetTrain/neg_acceptor_homo_train.txt',
    #     './SubsetHomo/SubsetTrain/pos_acceptor_homo_train.txt',
    # ]
    # test_dests = [
    #     './SubsetHomo/SubsetTest/neg_donor_homo_test.txt',
    #     './SubsetHomo/SubsetTest/pos_donor_homo_test.txt',
    #     './SubsetHomo/SubsetTest/neg_acceptor_homo_test.txt',
    #     './SubsetHomo/SubsetTest/pos_acceptor_homo_test.txt',
    # ]
    #
    # for index, target in enumerate(targets):
    #     with open(target, 'r') as f:
    #         lines = f.readlines()
    #         seed(123432)
    #         shuffle(lines)
    #         print(lines[:10])
    #         f.close()
    #     train = lines[:20000]
    #     test = lines[20000:24000]
    #     with open(train_dests[index], 'w') as f:
    #         f.writelines(train)
    #         f.close()
    #     with open(test_dests[index], 'w') as f:
    #         f.writelines(test)
    #         f.close()

if __name__ == '__main__':
    main()

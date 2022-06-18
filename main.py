import sys
import numpy as np
import itertools
import subprocess
import torch
import torch.nn as nn
import os, os.path
import pandas as pd
import pickle
import gzip

def get_prot_id_seq_dict_from_fasta_fl(fasta_fl_path):    
    prot_id_seq_dict = dict()
    prot_id = ""
    with open("{}".format(fasta_fl_path)) as f:
        for line in f:
            line = line.split("\n")[0]
            if line.startswith(">"):
                prot_id = line.split("|")[1]
                prot_id_seq_dict[prot_id] = ""
            else:
                prot_id_seq_dict[prot_id] = prot_id_seq_dict[prot_id] + line
    return prot_id_seq_dict

def get_all_aa_word_list(word_size):
    aa_list = get_aa_list()
    all_n_gram_list = list(itertools.product(aa_list, repeat=word_size))
    all_n_gram_list = [''.join(n_gram_tuple) for n_gram_tuple in all_n_gram_list]
    return all_n_gram_list

def get_aa_match_encodings():
    all_aa_matches = get_all_aa_word_list(2)
    aa_match_encoding_dict = dict()
    encod_int = 1
    for aa_pair in all_aa_matches:
        if aa_pair not in aa_match_encoding_dict.keys():
            aa_match_encoding_dict[aa_pair] = encod_int
            aa_match_encoding_dict[aa_pair[::-1]] = encod_int
            encod_int += 1
    return aa_match_encoding_dict

def get_aa_list():
    aa_list = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    return aa_list

def remove_nonstandard_aas(prot_seq):
    aa_list = get_aa_list()
    prot_seq_list = [aa for aa in prot_seq if aa in aa_list]
    prot_seq = ''.join(prot_seq_list)
    return prot_seq

def get_aa_match_encodings_generic(aaindex_encoding, full_matrix=False):
    aa_list = get_aa_list()
    encoding_fl = open("./Encodings/{}.txt".format(aaindex_encoding))
    lst_encoding_fl = encoding_fl.read().split("\n")
    encoding_fl.close()
    aa_match_encoding_dict = dict()
    starting_ind = -1
    for row_ind in range(len(lst_encoding_fl)-1):
        str_line = lst_encoding_fl[row_ind]
        if str_line.startswith("M rows"):
            starting_ind = row_ind + 1

        if  not str_line.startswith("//") and starting_ind != -1 and row_ind >= starting_ind:
            row_aa_ind = row_ind - starting_ind
            str_line = str_line.split(" ")

            while "" in str_line:
                str_line.remove("")
            for col_ind in range(len(str_line)):
                if not full_matrix:
                    aa_match_encoding_dict["{}{}".format(aa_list[row_aa_ind], aa_list[col_ind])] = round(float(str_line[col_ind]),3)
                    aa_match_encoding_dict["{}{}".format(aa_list[col_ind], aa_list[row_aa_ind])] = round(float(str_line[col_ind]),3)
                else:
                    aa_match_encoding_dict["{}{}".format(aa_list[row_aa_ind], aa_list[col_ind])] = round(
                        float(str_line[col_ind]), 3)
    return aa_match_encoding_dict

def get_sequence_matrix(seq, size, aaindex_encoding=None):
    aa_match_encoding_dict = None
    if aaindex_encoding == None:
        aa_match_encoding_dict = get_aa_match_encodings()
    elif aaindex_encoding == "ZHAC000103":
        aa_match_encoding_dict = get_aa_match_encodings_generic(aaindex_encoding, full_matrix=True)
    else:
        aa_match_encoding_dict = get_aa_match_encodings_generic(aaindex_encoding)
    seq = remove_nonstandard_aas(seq)
    lst = []
    for i in range(len(seq)):
        lst.append([])
        for j in range(len(seq)):
            lst[-1].append(aa_match_encoding_dict[seq[i] + seq[j]])

    torch_arr = torch.from_numpy(np.asarray(lst))
    size_of_tensor = torch_arr.shape[0]

    if size_of_tensor < size:
        padding_size = int((size - size_of_tensor) / 2)
        m = nn.ZeroPad2d(padding_size)
        if size_of_tensor % 2 != 0:
            m = nn.ZeroPad2d((padding_size, padding_size + 1, padding_size, padding_size + 1))
        torch_arr = m(torch_arr)
    else:
        torch_arr = torch_arr[:size, :size]
    return torch_arr


def save_separate_flattened_sequence_matrices_seperate(dataset_name, size, aaindex_encoding=None):
    fasta_fl_path = "./{}".format(dataset_name)
    prot_id_seq_dict = get_prot_id_seq_dict_from_fasta_fl(fasta_fl_path)
    count = 0

    for prot_id, seq in prot_id_seq_dict.items():

        count += 1
        if count % 100 == 0:
            print(count)

        for encoding in aaindex_encoding:
            if encoding == "2DEncodings":
                seq_torch_matrix = get_sequence_matrix(seq, size)
                np_matrix = seq_torch_matrix.numpy()
                df = pd.DataFrame(np_matrix)
                isim = str(np_matrix[9][2])
                save_folder = "./2DEncodingLEQ" + str(size) + "/"
                os.makedirs(os.path.dirname(save_folder), exist_ok=True)
                df.to_csv(save_folder + prot_id + ".csv", header = None, index=None)
            else:    
                seq_torch_matrix = get_sequence_matrix(seq, size, encoding)
                np_matrix = seq_torch_matrix.numpy()
                df = pd.DataFrame(np_matrix)
                isim = str(np_matrix[9][2])
                save_folder = "./"  + encoding + "LEQ" + str(size) + "/"
                os.makedirs(os.path.dirname(save_folder), exist_ok=True)
                df.to_csv( save_folder+ prot_id + ".csv", header = None, index=None)
    return


def save_separate_flattened_sequence_matrices_dict(dataset_name, size, aaindex_encoding=None):

    fasta_fl_path = "./{}".format(dataset_name)
    prot_id_seq_dict = get_prot_id_seq_dict_from_fasta_fl(fasta_fl_path)

    count = 0
    feature_dict = dict() # {Protein name:FeatureMatrix}
    for prot_id, seq in prot_id_seq_dict.items():
        count += 1
        if count % 100 == 0:
            print(count)

        encodings2stack = []
        for encoding in aaindex_encoding:

            if encoding == "2DEncodings":
                seq_torch_matrix = get_sequence_matrix(seq, size)
                flattened_seq_matrix_arr = np.array(seq_torch_matrix.contiguous().view(-1))
                encodings2stack.append(flattened_seq_matrix_arr)
            else:
                seq_torch_matrix = get_sequence_matrix(seq, size, encoding)
                flattened_seq_matrix_arr = np.array(seq_torch_matrix.contiguous().view(-1))
                encodings2stack.append(flattened_seq_matrix_arr)

        concat = np.vstack(encodings2stack)
        feat_vec = np.asarray(concat, dtype=float).reshape(len(encodings2stack), size, size)
        feature_dict[prot_id] = feat_vec


    with open("./" + dataset_name + '.pickle', 'wb') as f:
        pickle.dump(feature_dict, f, pickle.HIGHEST_PROTOCOL)

    #with gzip.open("./" + dataset_name + ".gz", "wb") as f:
        #pickle.dump(feature_dict, f, pickle.HIGHEST_PROTOCOL)


def get_aa_match_encodings_max_value(aaindex_encoding):
    import math
    encoding_fl = open(".\Encodings\{}.txt".format(aaindex_encoding))
    lst_encoding_fl = encoding_fl.read().split("\n")
    encoding_fl.close()
    starting_ind = -1
    max_value = -1000000000
    for row_ind in range(len(lst_encoding_fl)-1):
        str_line = lst_encoding_fl[row_ind]
        if str_line.startswith("M rows"):
            starting_ind = row_ind + 1

        if  not str_line.startswith("\\") and starting_ind != -1 and row_ind >= starting_ind:
            str_line = str_line.split(" ")

            while "" in str_line:
                str_line.remove("")

            for col_ind in range(len(str_line)):
                max_value = max(max_value, round(float(str_line[col_ind]),3))

    return max_value

def get_max_values_for_target_types(tar_feature_list):
    tar_feat_max_dict = dict()
    tar_feat_max_dict["sequencematrix500"] = 210.0
    tar_feat_max_dict["sequencematrix1000"] = 210.0
    for tar_feat in tar_feature_list[1:]:
        tar_feat = tar_feat.split("LEQ")[0]
        tar_feat_max_dict[tar_feat] = get_aa_match_encodings_max_value(tar_feat)

    return tar_feat_max_dict


def main():
    filename = sys.argv[1]
    form = sys.argv[2]
    matrixSize = int(sys.argv[3])
    encodings = sys.argv[4:]


    if form == "dict":
        save_separate_flattened_sequence_matrices_dict(filename, matrixSize, encodings)
    elif form == "seperate":
        save_separate_flattened_sequence_matrices_seperate(filename, matrixSize, encodings)
    else:
        return 31        


if __name__ == "__main__":
    main()

import sys
import os
import numpy as np

file_name = "ccf_qianyan_qm_result_A.csv"

path_1 = "./Checkpoints/MacBert_large/baseline_last2cls_seed_42_2080ti"
path_2 = "./Checkpoints/MacBert_large/baseline_last2cls_seed_42_1080ti"
path_3 = "./Checkpoints/MacBert_large/baseline_last2cls_seed_1000_1080ti"
path_4 = "./Checkpoints/MacBert_large/baseline_last2cls_seed_1000_2080ti"
path_5 = "./Checkpoints/MacBert_large/baseline_last2cls_seed_79_2080ti"
path_6 = "./Checkpoints/MacBert_large/baseline_last3cls_seed_42_2080ti"

def read_tsv(data_path):
    preds = None
    
    with open(data_path, 'r', encoding = 'utf-8') as f:
        for line in f:
            data = int(line)
            if preds is None:
                preds = np.array([data])
            else:
                preds = np.append(preds, [data])
    
    return preds            

def write_result(preds):

    with open(file_name, "w", encoding = "utf-8") as f:
        for pred in preds:
            f.write(str(pred) + "\n")

def main():
    path_a = os.path.join(path_1, file_name)
    path_b = os.path.join(path_2, file_name)
    path_c = os.path.join(path_3, file_name)
    path_d = os.path.join(path_4, file_name)
    path_e = os.path.join(path_5, file_name)
    path_f = os.path.join(path_6, file_name)
    
    preds_1 = read_tsv(path_a)
    preds_2 = read_tsv(path_b)
    preds_3 = read_tsv(path_c)
    preds_4 = read_tsv(path_d)
    preds_5 = read_tsv(path_e)
    preds_6 = read_tsv(path_f)

    preds = np.array((preds_1, preds_2, preds_3, preds_4, preds_5, preds_6))
    preds = preds.sum(axis = 0)
    preds[preds <= 3] = 0
    preds[preds >= 4] = 1
    #preds = preds.max(axis = 0)
    
    write_result(preds)
    

if __name__ == "__main__":
    main()




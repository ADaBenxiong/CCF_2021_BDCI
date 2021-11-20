import os
import sys
import numpy as np


data_path = "data/test_A.tsv"
label_path = "data/pseudo_labels/ccf_qianyan_qm_result_A.csv"
train_path = "data/pseudo_label.txt"

def read_text_pair(data_path, label_path):
    
    examples = []
    with open(data_path, 'r', encoding = 'utf-8') as f:
        
        for line in f:
            data = line.rstrip().split("\t")
            if len(data) != 2:
                continue
            examples.append((data[0], data[1]))

    labels = []
    with open(label_path, 'r', encoding = 'utf-8') as lf:
        
        for line in lf:
            labels.append(line)
    
    assert len(examples) == len(labels)
    
    #if os.path.exists(train_path) == False:
        
    with open(train_path, 'w', encoding = 'utf-8') as fw:
        
        for num in range(len(examples)):
            fw.write(examples[num][0] + "\t" + examples[num][1] + "\t" + labels[num])    
        
if __name__ == "__main__":
    read_text_pair(data_path, label_path)









import numpy as np

train_data_path = "data/train.txt"
dev_data_path = "data/dev.txt"

def analyze():
    pos_count = 0
    neg_count = 0
    with open(train_data_path, 'r', encoding = 'utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if data[-1] == '1':
                pos_count += 1
            else:
                neg_count += 1
    print("train set pos(1) proportion:" + str(pos_count / (pos_count + neg_count)))

    pos_count = 0
    neg_count = 0
    with open(dev_data_path, 'r', encoding = 'utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if data[-1] == '1':
                pos_count += 1
            else:
                neg_count += 1
    print("dev set pos(1) proportion:" + str(pos_count / (pos_count + neg_count)))
                

if __name__ == "__main__":
    analyze()



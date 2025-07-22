import torch
import collections
import pdb
import torch.utils.data
import csv
import json
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import math
import numpy as np
import os

def loadData(filename, windows, n_hms):
    with open(filename) as fi:
        csv_reader=csv.reader(fi)
        data=list(csv_reader)
    fi.close()
    
    nrows=len(data)
    ngenes = nrows / windows
    print(f"Number of genes: {ngenes:.0f} in {filename}")
    print(f"Number of entries: {nrows}")
    print(f"Number of HMs: {n_hms}")

    attr=collections.OrderedDict()
    for i in range(0, nrows, windows):
        hms_tensors = []
        for h in range(n_hms):
            hms_tensors.append(torch.zeros(windows, 1))

        for w in range(0, windows):
            for h in range(n_hms):
                hms_tensors[h][w][0] = int(data[i+w][h + 2])
        
        geneID = str(data[i][0].split("_")[0])
        thresholded_expr = int(data[i+windows-1][7])
        
        attr[i // windows] = {
            'geneID': geneID,
            'expr': thresholded_expr,
            'hms': hms_tensors
        }
    return attr

class HMData(Dataset):
    def __init__(self, data_cell, transform=None):
        self.c1 = data_cell
    def __len__(self):
        return len(self.c1)
    def __getitem__(self, i):
        final_data_c1 = torch.cat(self.c1[i]['hms'], 1)
        label = self.c1[i]['expr']
        geneID = self.c1[i]['geneID']
        sample = {'geneID': geneID, 'input': final_data_c1, 'label': label}
        return sample

def load_data(args):
    all_train_datasets = []
    all_valid_datasets = []
    all_test_datasets = []
    
    for cell_type in args.cell_types:
        print(f"==> Loading data for cell type: {cell_type}")
        
        train_file = os.path.join(args.data_root, cell_type, "classification", "train.csv")
        train_data = loadData(train_file, args.n_bins, args.n_hms)
        all_train_datasets.append(HMData(train_data))

        valid_file = os.path.join(args.data_root, cell_type, "classification", "valid.csv")
        valid_data = loadData(valid_file, args.n_bins, args.n_hms)
        all_valid_datasets.append(HMData(valid_data))

        test_file = os.path.join(args.data_root, cell_type, "classification", "test.csv")
        test_data = loadData(test_file, args.n_bins, args.n_hms)
        all_test_datasets.append(HMData(test_data))

    combined_train_dataset = torch.utils.data.ConcatDataset(all_train_datasets)
    combined_valid_dataset = torch.utils.data.ConcatDataset(all_valid_datasets)
    combined_test_dataset = torch.utils.data.ConcatDataset(all_test_datasets)

    print(f"Total number of training genes: {len(combined_train_dataset)}")
    print(f"Total number of validation genes: {len(combined_valid_dataset)}")
    print(f"Total number of testing genes: {len(combined_test_dataset)}")
    
    Train = torch.utils.data.DataLoader(combined_train_dataset, batch_size=args.batch_size, shuffle=True)
    Valid = torch.utils.data.DataLoader(combined_valid_dataset, batch_size=args.batch_size, shuffle=False)
    Test = torch.utils.data.DataLoader(combined_test_dataset, batch_size=args.batch_size, shuffle=False)

    return Train, Valid, Test

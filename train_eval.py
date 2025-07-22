import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import cuda
import sys, os
import random
import numpy as np
import csv
from sklearn import metrics
import models as Model
import data

# --- Helper Functions for Metric Computation ---
def compute_aupr(all_targets, all_predictions):
    aupr_array = []
    for i in range(all_targets.shape[1]):
        try:
            precision, recall, thresholds = metrics.precision_recall_curve(all_targets[:,i], all_predictions[:,i], pos_label=1)
            auPR = metrics.auc(recall, precision)
            if not math.isnan(auPR):
                aupr_array.append(np.nan_to_num(auPR))
        except: 
            pass
    
    aupr_array = np.array(aupr_array)
    return np.mean(aupr_array), np.median(aupr_array), np.var(aupr_array), aupr_array

def compute_auc(all_targets, all_predictions):
    auc_array = []
    for i in range(all_targets.shape[1]):
        try: 
            auROC = metrics.roc_auc_score(all_targets[:,i], all_predictions[:,i])
            auc_array.append(auROC)
        except ValueError:
            pass
    
    auc_array = np.array(auc_array)
    return np.mean(auc_array), np.median(auc_array), np.var(auc_array), auc_array

def compute_metrics(predictions, targets):
    pred = predictions.numpy()
    targets = targets.numpy()
    mean_auc, _, _, _ = compute_auc(targets, pred)
    mean_aupr, _, _, _ = compute_aupr(targets, pred)
    return mean_aupr, mean_auc

# --- Training and Testing Functions ---

def train(model, TrainData, optimizer, args, dtype):
    model.train()
    
    diff_targets = torch.zeros(TrainData.dataset.__len__(), 1)
    predictions = torch.zeros(diff_targets.size(0), 1)
    per_epoch_loss = 0
    
    for idx, Sample in enumerate(TrainData):
        start, end = (idx * args.batch_size), min((idx * args.batch_size) + args.batch_size, TrainData.dataset.__len__())
        inputs_1 = Sample['input']
        batch_diff_targets = Sample['label'].unsqueeze(1).float()
        
        optimizer.zero_grad()
        batch_predictions = model(inputs_1.type(dtype))
        loss = F.binary_cross_entropy_with_logits(batch_predictions.cpu(), batch_diff_targets, reduction='mean')
        
        per_epoch_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        
        diff_targets[start:end, 0] = batch_diff_targets[:, 0]
        batch_predictions = torch.sigmoid(batch_predictions)
        predictions[start:end] = batch_predictions.data.cpu()
        
    per_epoch_loss = per_epoch_loss / len(TrainData)
    return predictions, diff_targets, per_epoch_loss


def test_model(model, TestData, args, dtype):
    model.eval()
    
    diff_targets = torch.zeros(TestData.dataset.__len__(), 1)
    predictions = torch.zeros(diff_targets.size(0), 1)
    
    per_epoch_loss = 0
    with torch.no_grad():
        for idx, Sample in enumerate(TestData):
            start, end = (idx * args.batch_size), min((idx * args.batch_size) + args.batch_size, TestData.dataset.__len__())
            inputs_1 = Sample['input']
            batch_diff_targets = Sample['label'].unsqueeze(1).float()
            
            batch_predictions = model(inputs_1.type(dtype))
            loss = F.binary_cross_entropy_with_logits(batch_predictions.cpu(), batch_diff_targets, reduction='mean')
            
            diff_targets[start:end, 0] = batch_diff_targets[:, 0]
            batch_predictions = torch.sigmoid(batch_predictions)
            predictions[start:end] = batch_predictions.data.cpu()
            per_epoch_loss += loss.item()

    per_epoch_loss = per_epoch_loss / len(TestData)
    return predictions, diff_targets, per_epoch_loss

# --- Main Execution Block ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Train and Evaluate Script')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--model_type', type=str, default='bilstm_attention', help='Model architecture')
    parser.add_argument('--clip', type=float, default=1, help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=30, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout applied to layers')
    parser.add_argument('--cell_type', type=str, default='E003', help='cell type')
    parser.add_argument('--save_root', type=str, default='./Results/', help='where to save')
    parser.add_argument('--data_root', type=str, default='./data/', help='data location')
    parser.add_argument('--gpuid', type=int, default=0, help='CUDA gpu')
    parser.add_argument('--gpu', type=int, default=0, help='CUDA gpu')
    parser.add_argument('--n_hms', type=int, default=5, help='number of histone modifications')
    parser.add_argument('--n_bins', type=int, default=100, help='number of bins')
    parser.add_argument('--bin_rnn_size', type=int, default=32, help='bin rnn size')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--unidirectional', action='store_true', help='bidirectional/undirectional LSTM')
    parser.add_argument('--save_attention_maps', action='store_true', help='set to save attention maps')
    parser.add_argument('--attentionfilename', type=str, default='beta_attention.txt', help='where to save attnetion maps')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads in the transformer')
    parser.add_argument('--dim_feedforward', type=int, default=256, help='Feedforward dimension in the transformer')
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], help='Script mode: "train" or "eval"')
    parser.add_argument('--cell_types', nargs='+', default=['E003'], help='List of cell types to use for training/evaluation')

    args = parser.parse_args()

    torch.manual_seed(1)
    model_name = f'{args.cell_type}_{args.model_type}'
    args.bidirectional = not args.unidirectional

    print(f"Running in '{args.mode}' mode.")

    args.data_root = os.path.join(args.data_root)
    args.save_root = os.path.join(args.save_root, args.cell_type)
    model_dir = os.path.join(args.save_root, model_name)
    
    # Load data for both modes
    Train, Valid, Test = data.load_data(args)

    if args.mode == 'train':
        print('==> Starting Training')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model = Model.att_chrome(args)
        if torch.cuda.device_count() > 0:
            torch.cuda.manual_seed_all(1)
            dtype = torch.cuda.FloatTensor
            model.type(dtype)
        else:
            dtype = torch.FloatTensor
        
        print("==> Initializing a new model")
        for p in model.parameters():
            if p.requires_grad:
                p.data.uniform_(-0.1, 0.1)
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_valid_avgAUC = -1
        for epoch in range(0, args.epochs):
            print(f"---------------------------------------- Training Epoch {epoch+1} -----------------------------------")
            predictions, diff_targets, train_loss = train(model, Train, optimizer, args, dtype)
            train_avgAUPR, train_avgAUC = compute_metrics(predictions, diff_targets)

            predictions, diff_targets, valid_loss = test_model(model, Valid, args, dtype)
            valid_avgAUPR, valid_avgAUC = compute_metrics(predictions, diff_targets)

            if valid_avgAUC >= best_valid_avgAUC:
                best_valid_avgAUC = valid_avgAUC
                torch.save(model.cpu().state_dict(), f"{model_dir}/{model_name}_avgAUC_model.pt")
                model.type(dtype)

            print(f"Epoch: {epoch} | train AUC: {train_avgAUC:.4f} | valid AUC: {valid_avgAUC:.4f} | best valid AUC: {best_valid_avgAUC:.4f}")

        print("\nFinished training. Best model saved to:", model_dir)

    elif args.mode == 'eval':
        print('==> Starting Evaluation')
        model = Model.att_chrome(args)
        if torch.cuda.device_count() > 0:
            dtype = torch.cuda.FloatTensor
            model.type(dtype)
        else:
            dtype = torch.FloatTensor
        
        model_path = f"{model_dir}/{model_name}_avgAUC_model.pt"
        if os.path.exists(model_path):
            print(f"==> Loading saved model from {model_path}")
            model.load_state_dict(torch.load(model_path))
            model.eval()
        else:
            print(f"Error: Model file not found at {model_path}. Please train a model first.")
            sys.exit()

        predictions, diff_targets, test_loss = test_model(model, Test, args, dtype)
        test_avgAUPR, test_avgAUC = compute_metrics(predictions, diff_targets)

        print(f"Final Test AUC: {test_avgAUC:.4f} | Final Test AUPR: {test_avgAUPR:.4f}")
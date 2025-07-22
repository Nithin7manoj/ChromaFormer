
# ChromaFormer: A Hybrid Deep Learning Framework for Epigenetic Prediction

### 1. Introduction

This project introduces **ChromaFormer**, a novel deep learning framework designed for predicting gene expression states from complex patterns of histone modifications. Building upon the foundational ideas of hierarchical recurrent neural networks and attention mechanisms (inspired by **AttentiveChrome**), ChromaFormer integrates the powerful Transformer architecture to enhance predictive accuracy and model generalization.

Our work provides a comprehensive comparative analysis of three distinct deep learning models, highlighting the benefits of a hybrid approach when trained on diverse biological datasets from multiple cell lines.

### 2. Key Features

- **Advanced Model Architectures:** Implementation and comparison of BiLSTM+Attention, BiLSTM+Transformer, and a novel BiLSTM+Attention+Transformer hybrid model.
- **Scalable Data Handling:** A robust data loading pipeline capable of aggregating and processing epigenetic data from numerous cell lines to improve model generalization.
- **Performance Benchmarking:** Rigorous evaluation of models using standard classification metrics (AUC, AUPR) to quantify improvements.
- **Modular Codebase:** A clean and organized project structure, separating concerns into dedicated modules for models, data processing, and training/evaluation.

### 3. Model Architectures

The framework employs a hierarchical modeling approach:

1. **BiLSTM + Attention (Baseline):** This model uses Bidirectional LSTMs to process sequential genomic bin data for each histone modification. An attention mechanism then aggregates these bin-level features to form a comprehensive representation for each histone mark. A final attention layer combines these mark-level representations for prediction.
2. **BiLSTM + Transformer (Novel):** Similar to the baseline, BiLSTMs process bin-level data. However, instead of a traditional attention layer, a Transformer Encoder is utilized at the higher level to capture intricate, long-range dependencies and global context across the different histone modifications.
3. **Hybrid Model (BiLSTM + Attention + Transformer):** This is our primary contribution. It combines the strengths of both attention and Transformer mechanisms. After initial BiLSTM processing, both a standard attention layer and a Transformer Encoder process the histone mark representations. Their outputs are then fused (concatenated) before the final prediction layer, aiming for a more robust and comprehensive model.

### 4. Results & Analysis

Our experiments, conducted on a combined dataset of **10 distinct cell lines**, demonstrate a significant improvement in predictive performance with the hybrid architecture.

**Performance Comparison (AUC)**

<img width="1000" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/76db21d6-f691-4924-99a4-755d02f2bf69" />


### 5. Project Setup

**Prerequisites**

To run this project, you need Python 3.8+ installed. All required dependencies can be installed via pip:

```
pip install torch numpy scipy scikit-learn pandas matplotlib
```

**Directory Structure**

Ensure your project directory is organized as follows. The `data/` folder should contain subdirectories for each cell type you intend to use, with `train.csv`, `valid.csv`, and `test.csv` files within their respective `classification/` subfolders.

```
your_project_folder/
├── train_and_eval.py
├── models.py
├── data.py
├── .gitignore
└── data/
    └── E003/
        └── classification/
            ├── train.csv
            ├── valid.csv
            └── test.csv
    └── E005/
        └── classification/
            ├── train.csv
            ...
    └── [Other Cell Type Folders]/
        └── classification/
            ├── train.csv
            ...

```

### 6. Dataset

The dataset utilized in this project comprises gene expression profiles and corresponding histone modification data across various human cell lines. The rows are bins for all genes (100 rows per gene) and the columns are organised as follows:

GeneID, Bin ID, H3K27me3 count, H3K36me3 count, H3K4me1 count, H3K4me3 count, H3K9me3 counts, Binary Label for gene expression (0/1)
e.g. 000003,1,4,3,0,8,4,1
**Link to Dataset:** *https://zenodo.org/record/2652278*

### 7. How to Run the Code

The `train_and_eval.py` script serves as the central command-line interface for both training and evaluating the models.

**To train and save a model on a single cell line (e.g., E003):**

```bash
python train_and_eval.py --mode train --model_type bilstm_attention --cell_types E003
```

**To train and save a model on a combined dataset (e.g., 10 cell lines):**

```bash
python train_and_eval.py --mode train --model_type bilstm_attention_transformer --cell_types E003 E005 E007 E012 E016 E027 E037 E047 E053 E055 --n_heads 4 --dim_feedforward 256
```

**To evaluate a previously saved model on its corresponding test set:**

```bash
python train_and_eval.py --mode eval --model_type bilstm_attention_transformer --cell_types E003 E005 E007 E012 E016 E027 E037 E047 E053 E055 --n_heads 4 --dim_feedforward 256
```

### 8. License

This project is licensed under the MIT License. For more details, please refer to the `LICENSE` file in the repository.

### 9. Acknowledgments & References

This project is inspired by and builds upon the foundational work presented in the original **AttentiveChrome** paper. We extend its concepts by introducing a hybrid architecture and conducting a comprehensive comparative analysis.

**Original Paper:**

Singh R, Lanchantin J, Sekhon A, Qi Y. Attend and Predict: Understanding Gene Regulation by Selective Attention on Chromatin. Adv Neural Inf Process Syst. 2017 Dec;30:6785-6795. PMID: 30147283; PMCID: PMC6105294.

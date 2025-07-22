import matplotlib.pyplot as plt
import numpy as np

# Replace these placeholder values with your actual results from Step 1
# This dictionary holds the test AUC for each model in two scenarios.
results = {
    'BiLSTM+Attention': {
        '1 Dataset': 0.7899,  # Replace with your single-dataset AUC
        '10 Datasets': 0.8053  # Replace with your 10-dataset AUC
    },
    'BiLSTM+Transformer': {
        '1 Dataset': 0.7849,  # Replace with your single-dataset AUC
        '10 Datasets': 0.7959  # Replace with your 10-dataset AUC
    },
    'BiLSTM+Attn+Tfmr': {
        '1 Dataset': 0.7878,  # Replace with your single-dataset AUC
        '10 Datasets': 0.8044  # Replace with your 10-dataset AUC
    }
}

model_types = list(results.keys())
dataset_types = list(results[model_types[0]].keys())
bar_width = 0.35
x = np.arange(len(model_types))

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - bar_width/2, [results[m][dataset_types[0]] for m in model_types], bar_width, label=dataset_types[0], color='#4CAF50')
bars2 = ax.bar(x + bar_width/2, [results[m][dataset_types[1]] for m in model_types], bar_width, label=dataset_types[1], color='#2196F3')

ax.set_ylabel('Final Test AUC')
ax.set_title('Model Performance: 1 Dataset vs. 10 Datasets')
ax.set_xticks(x)
ax.set_xticklabels(model_types)
ax.legend()

def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)

plt.show()

# To save the plot instead of showing it
# plt.savefig('model_comparison.png', dpi=300)
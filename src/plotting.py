import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt

def plot_roc(y, y_hats, labels, x_max=1.0):
    fig, ax = plt.subplots()
    for y_hat, label in zip(y_hats, labels):
        auc = roc_auc_score(y, y_hat)
        acc = accuracy_score(y, y_hat>0.5)
        fpr, tpr, thresh = roc_curve(y, y_hat)
        ax.plot(fpr, tpr, label=f'{label}; AUC={auc:.3f}; ACC={acc:.3f}', marker='o', markersize=1)
    ax.legend()
    ax.grid()
    ax.plot(np.linspace(0, 1, 20), np.linspace(0, 1, 20), linestyle='--')
    ax.set_title('ROC curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_xlim([-0.01, x_max])
    _ = ax.set_ylabel('True Positive Rate')

def plot_losses(train_losses, test_losses):
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='Train')
    ax.plot(test_losses, label='Test')
    ax.grid()
    ax.legend()

def compare_histograms(X_1, X_2, x1_labels, x2_labels, logx=False):
    fig, axs = plt.subplots(X_1.shape[1], 1, figsize=(4, 4*X_1.shape[1]))
    kwargs = dict(alpha=0.5, density=True)
    if logx:
        X_1 = np.log10(X_1)
        X_2 = np.log10(X_2)

    X = np.vstack([X_1, X_2])
    if not hasattr(axs, '__iter__'):
        axs = [axs]
    for idx, ax in enumerate(axs):
        bins = np.linspace(X[:,idx].min(), X[:,idx].max(), 50)
        _, _, _ = ax.hist(X_1[:,idx], bins=bins, label=f'{"log " if logx else ""}X_1: {x1_labels[idx]}', **kwargs)
        _, _, _ = ax.hist(X_2[:,idx], bins=bins, label=f'{"log " if logx else ""}X_2: {x2_labels[idx]}', **kwargs)
        ax.legend()

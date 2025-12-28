import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns


class PlotUtility:    
    @staticmethod
    def plot_training_history(history_file: str, output_path='plots/triplet_history.png'):
        """ Plot and save training and validation loss, and learning rate schedule. """
        
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        _, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # plot loss
        axes[0].plot(history['epoch'], history['loss'], label='Train', linewidth=2)
        axes[0].plot(history['epoch'], history['val_loss'], label='Val', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Triplet Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # plot learning rate
        axes[1].plot(history['epoch'], history['lr'], linewidth=2, color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved training history plot at: {output_path}")
        
    
    @staticmethod
    def plot_embedings_separation(
        same_distributions: list,
        diff_distributions: list,
        best_thresh: float,
        output_path='plots/embeddings_separation.png'
    ):
    
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(
            same_distributions,
            bins=50, 
            alpha=0.6, 
            label='Same (1)', 
            color='blue', 
            density=True
        )
        
        ax.hist(
            diff_distributions,
            bins=50, 
            alpha=0.6, 
            label='Different (0)', 
            color='orange', 
            density=True
        )
        
        ax.axvline(
            best_thresh,
            color='red', 
            linestyle='--', 
            linewidth=2, 
            label=f'Threshold={best_thresh:.3f}'
        )
        
        ax.set_xlabel('Distance')
        ax.set_ylabel('Density')
        ax.set_title('Distance Distribution by Class')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved embeddings separation plot at: {output_path}")
        
        
    @staticmethod
    def plot_confusion_matrix(
        predictions: list,
        labels: list,
        output_path='plots/triple_confusion_matrix.png'
    ):  
        cm = confusion_matrix(labels, predictions)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues', 
            xticklabels=['Pred 0', 'Pred 1'], 
            yticklabels=['True 0', 'True 1']
        )
        
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved confusion matrix plot at: {output_path}")
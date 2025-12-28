from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import gc
from tqdm import tqdm
import numpy as np

from utils.data_loader import TripletDataLoader
from encoder.siamese_model import SiameseModel
from utils.util import HistoryCallback, triplet_loss
from utils.plot_utility import PlotUtility


CONFIG = {
    'embedding_dim': 192,
    'base_filters': 32,
    'layers_filters_multiplier': [1, 2, 4, 8],
    'batch_size': 32,
    'learning_rate': 0.0003,
    'epochs': 60,
    'early_stopping_patience': 12,
    'dropout_rate': 0.25,
    'l2_reg': 0.00005,
    'augment': True,
    'aug_prob': 0.5,
    'mixed_precision': True,
    'margin': 0.5,
}


class TripletSiameseTrainer:
    def __init__(
        self,
        data_dir: str,
        samples_dir: str,
        img_shape=(256, 256, 1),
        config=None
    ):
        self.data_dir = Path(data_dir)
        self.samples_dir = Path(samples_dir)
        self.img_shape = img_shape
        self.config = config or CONFIG
        self.base_network = None
        self.triplet_model = None
        self.best_threshold = 0.7
        
        if self.config['mixed_precision']:
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
            print("[TripletSiamese] Mixed precision enabled")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    
    
    ################################################################
    #                     TRAINING METHODS                      #
    ################################################################
                
    def load_data(self):
        """ Load training, validation, and test data from CSV files."""
        self.train_df = pd.read_csv(self.data_dir / 'train.csv')
        self.val_df = pd.read_csv(self.data_dir / 'validation.csv')
        self.test_df = pd.read_csv(self.data_dir / 'test.csv')
        
        
    def train(self) -> keras.callbacks.History:
        """ Train the triplet Siamese model. """
        
        # build outputs directories
        Path('models').mkdir(exist_ok=True)
        Path('plots').mkdir(exist_ok=True)
        
        self._build_model()
        
        train_gen, val_gen = self._get_data_loaders()
        
        callbacks = self._get_training_callbacks()
        
        self.triplet_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss=triplet_loss(margin=self.config['margin'])
        )
        
        history = self.triplet_model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=self.config['epochs'],
            callbacks=list(callbacks.values()),
            verbose=2
        )   
        
        del train_gen
        del val_gen
        gc.collect()
        
        return history 
        
        
    def _get_data_loaders(self) -> tuple:
        """ Create data loaders for training and validation datasets. """
        
        train_gen = TripletDataLoader(
            df=self.train_df, 
            samples_dir=self.samples_dir,
            batch_size=self.config['batch_size'],
            shuffle=True,
            augment=self.config['augment'],
            aug_prob=self.config['aug_prob']
        )
        
        val_gen = TripletDataLoader(
            df=self.val_df, 
            samples_dir=self.samples_dir,
            batch_size=self.config['batch_size'],
            shuffle=False,
            augment=False
        )
        
        return train_gen, val_gen
        
    
    def _build_model(self):
        """ Build the Siamese triplet model. """
        
        siamese = SiameseModel(
            img_shape=self.img_shape,
            embedding_dim=self.config['embedding_dim'],
            base_filters=self.config['base_filters'],
            layers_filters_multiplier=self.config['layers_filters_multiplier'],
            dropout=self.config['dropout_rate'],
            l2_reg=self.config['l2_reg']
        )
        
        self.base_network, self.triplet_model = siamese.build_model()
        
        self.triplet_model.summary()
        
    
    def _get_training_callbacks(self) -> dict:
        """ Create training callbacks: EarlyStopping, ReduceLROnPlateau, and HistoryCallback. """
        
        early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            )
        
        reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        
        history_logger = HistoryCallback(log_file='models/triplet_history.json')
        
        return {
            'early_stopping': early_stopping,
            'reduce_lr': reduce_lr,
            'history_logger': history_logger
        }
        
    
    ################################################################
    #                  CLASSIFICATION METHODS                      #
    ################################################################
    
    def find_optimal_threshold(self) -> tuple:
        """ 
        Find the optimal threshold on the validation set to split embeddings.
        
        Returns a tuple containing:
            - best_thresh (float): The best threshold value.
            - best_acc (float): The best accuracy achieved with the best threshold.
        """
        
        all_distances, all_labels = self.__compute_embeddings_distances()
        
        same_dist = all_distances[all_labels == 1]
        diff_dist = all_distances[all_labels == 0]
        
        print(f"\nDistance Statistics:")
        print(f"  Same pairs:      {same_dist.mean():.4f} ± {same_dist.std():.4f}")
        print(f"  Different pairs: {diff_dist.mean():.4f} ± {diff_dist.std():.4f}")
        print(f"  Separation:      {diff_dist.mean() / (same_dist.mean() + 1e-8):.2f}x")
        
        best_acc = 0.0
        best_thresh = 0.0
        
        thresholds = np.linspace(all_distances.min(), all_distances.max(), num=300)
        
        for threshold in thresholds:
            preds = (all_distances < threshold).astype(int)
            acc = (preds == all_labels).mean()
            
            if acc > best_acc:
                best_acc = acc
                best_thresh = threshold
                
        print(f"\nOptimal threshold found: {best_thresh:.4f} with accuracy: {best_acc:.4f}")
        
        PlotUtility.plot_embedings_separation(
            same_distributions=same_dist,
            diff_distributions=diff_dist,
            best_thresh=best_thresh,
            output_path='plots/embeddings_separation.png'
        )
        
        PlotUtility.plot_confusion_matrix(
            predictions=preds,
            labels=all_labels,
            output_path='plots/triple_confusion_matrix.png'
        )
        
        self.best_threshold = best_thresh
        return best_thresh, best_acc
        
        
    def __compute_embeddings_distances(self) -> tuple:
        """
        Compute embeddings distances for all pairs in the validation set.
        
        Returns a tuple containing:
            - all_distances (np.ndarray): List of distances between embeddings.
            - all_labels (np.ndarray): Corresponding labels (1 for same, 0 for different
        """
        
        all_distances = []
        all_labels = []
    
        batch_size = 64
        
        for i in tqdm(range(0, len(self.val_df), batch_size), desc="Computing distances"):
            batch_df = self.val_df.iloc[i:i+batch_size]
            
            img1_list, img2_list, labels = [], [], []
            
            for _, row in batch_df.iterrows():
                img1_path = self.samples_dir / f"{row['id_noise_1']}.npy"
                img2_path = self.samples_dir / f"{row['id_noise_2']}.npy"
                
                img1 = np.load(img1_path).astype(np.float32)
                img2 = np.load(img2_path).astype(np.float32)
                
                img1 = (img1 - img1.mean()) / (img1.std() + 1e-8)
                img2 = (img2 - img2.mean()) / (img2.std() + 1e-8)
                
                img1_list.append(img1[..., np.newaxis])
                img2_list.append(img2[..., np.newaxis])
                labels.append(row['label'])
            
            # computing embeddings
            emb1 = self.base_network.predict(np.array(img1_list), verbose=0)
            emb2 = self.base_network.predict(np.array(img2_list), verbose=0)
            
            distances = np.linalg.norm(emb1 - emb2, axis=1)
            
            all_distances.extend(distances)
            all_labels.extend(labels)
            
        all_distances = np.array(all_distances)
        all_labels = np.array(all_labels)
        
        return all_distances, all_labels
    
    
    ################################################################
    #                     EVALUATION METHODS                      #
    ################################################################
    
    def evaluate(self, output_file='triplet_test_predictions.csv'):
        """ 
        Evaluate the model on the test dataset using the optimal threshold.
        Writes the labels to a CSV file.
        """
        
        all_preds = self._get_test_predictions()
        
        id_pairs = [
            f"({row['id_noise_1']},{row['id_noise_2']})"
            for _, row in self.test_df.iterrows()
        ]
        
        submission = pd.DataFrame({
            'id_pair': id_pairs,
            'label': all_preds[:len(id_pairs)]
        })
        
        submission.to_csv(output_file, index=False)
        print(f"Test predictions saved to: {output_file}")
        
    
    def _get_test_predictions(self) -> list:
        """
        Get predictions for the test dataset.
        
        Returns:
            - all_preds (list): List of predicted labels for the test set.
        """
        
        all_preds = []
        
        batch_size = 64
        for i in tqdm(range(0, len(self.test_df), batch_size), desc="Predicting"):
            batch_df = self.test_df.iloc[i:i+batch_size]
            
            img1_list, img2_list = [], []
            
            for _, row in batch_df.iterrows():
                img1 = np.load(self.samples_dir / f"{row['id_noise_1']}.npy").astype(np.float32)
                img2 = np.load(self.samples_dir / f"{row['id_noise_2']}.npy").astype(np.float32)
                
                img1 = (img1 - img1.mean()) / (img1.std() + 1e-8)
                img2 = (img2 - img2.mean()) / (img2.std() + 1e-8)
                
                img1_list.append(img1[..., np.newaxis])
                img2_list.append(img2[..., np.newaxis])
            
            emb1 = self.base_network.predict(np.array(img1_list), verbose=0)
            emb2 = self.base_network.predict(np.array(img2_list), verbose=0)
            
            distances = np.linalg.norm(emb1 - emb2, axis=1)
            preds = (distances < self.best_threshold).astype(int)
            all_preds.extend(preds)
            
        return all_preds
       
    
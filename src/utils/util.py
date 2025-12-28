import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import Callback
import json

def triplet_loss(margin=0.5):

    def loss_fn(y_true, y_pred):
        # y_pred shape: (batch, embedding_dim * 3)
        embedding_dim = tf.shape(y_pred)[1] // 3
        
        anchor = y_pred[:, :embedding_dim]
        positive = y_pred[:, embedding_dim:embedding_dim*2]
        negative = y_pred[:, embedding_dim*2:]
        
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        
        loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
        
        return tf.reduce_mean(loss)
    
    return loss_fn


class HistoryCallback(Callback):
    def __init__(self, log_file='models/triplet_history.json'):
        super().__init__()
        self.log_file = log_file
        self.history_data = {
            'epoch': [],
            'loss': [],
            'val_loss': [],
            'lr': []
        }
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history_data['epoch'].append(epoch)
        self.history_data['loss'].append(float(logs.get('loss', 0)))
        self.history_data['val_loss'].append(float(logs.get('val_loss', 0)))
        self.history_data['lr'].append(float(keras.backend.get_value(self.model.optimizer.lr)))
        
        with open(self.log_file, 'w') as f:
            json.dump(self.history_data, f, indent=2)
import keras
import numpy as np
from pathlib import Path
from pandas import DataFrame

class TripletDataLoader(keras.utils.Sequence):
    def __init__(
        self,
        df: DataFrame, 
        samples_dir: str,
        batch_size=32,
        shuffle=True, 
        augment=False, 
        aug_prob=0.5
    ):
        self.df = df.reset_index(drop=True)
        self.samples_dir = Path(samples_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.aug_prob = aug_prob
        
        self.same_pairs = df[df['label'] == 1].reset_index(drop=True)
        self.diff_pairs = df[df['label'] == 0].reset_index(drop=True)
        
        self.on_epoch_end()
    
    def __len__(self):
        return len(self.same_pairs) // self.batch_size
    
    
    def load_and_process(self, img_id: str):
        """ 
        Load and image by its ID, normalize it, and apply augmentations if needed.
        """
        
        img_path = self.samples_dir / f"{img_id}.npy"
        img = np.load(img_path).astype(np.float32)
        
        img = (img - img.mean()) / (img.std() + 1e-8)
        
        should_augment = self.augment and np.random.rand() < self.aug_prob
        if should_augment:
            if np.random.rand() > 0.5:
                img = np.fliplr(img)
                
            k = np.random.randint(0, 4)
            if k > 0:
                img = np.rot90(img, k)
        
        return img[..., np.newaxis]
    
    
    def __getitem__(self, index) -> tuple:
        """
        Generate a new batch of triplet data.
        
        The pairs are selected such that for each anchor-positive pair,
        a random negative sample is chosen from the different pairs.
        
        Returns a tuple containing:
        - A list of three numpy arrays: anchors, positives, negatives.
        - A numpy array of dummy labels (zeros).
        """
        anchors, positives, negatives = [], [], []
        
        for i in range(self.batch_size):
            idx = (index * self.batch_size + i) % len(self.same_pairs)
            
            # get positive pair
            pos_row = self.same_pairs.iloc[idx]
            anchor = self.load_and_process(pos_row['id_noise_1'])
            positive = self.load_and_process(pos_row['id_noise_2'])
            
            # get random negative
            neg_idx = np.random.randint(len(self.diff_pairs))
            neg_row = self.diff_pairs.iloc[neg_idx]
            negative = self.load_and_process(neg_row['id_noise_2'])
            
            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)
        
        return [
            np.array(anchors),
            np.array(positives),
            np.array(negatives)
        ], np.zeros(self.batch_size)  # Dummy labels
    
    
    def on_epoch_end(self):
        if self.shuffle:
            self.same_pairs = self.same_pairs.sample(frac=1).reset_index(drop=True)
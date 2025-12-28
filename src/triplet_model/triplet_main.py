from triplet_siamese_trainer import TripletSiameseTrainer
from time import time

from utils.plot_utility import PlotUtility

def main():
    data_dir = '../data'
    samples_dir = '../data/samples'
    
    config = {
        'embedding_dim': 384,
        'base_filters': 2,
        'batch_size': 32,
        'layers_filters_multiplier': [1, 2, 4, 8],
        'learning_rate': 0.00075,
        'epochs': 10,
        'early_stopping_patience': 15,
        'dropout_rate': 0.33,
        'l2_reg': 0.0001,
        'augment': True,
        'aug_prob': 0.5,
        'mixed_precision': True,
        'margin': 0.75,
    }
    
    trainer = TripletSiameseTrainer(
        data_dir=data_dir,
        samples_dir=samples_dir,
        img_shape=(256, 256, 1),
        config=config
    )
    
    print("Loading data...")
    trainer.load_data()
    print(f"Loaded {len(trainer.train_df)} training and {len(trainer.val_df)} validation samples.")
    
    start_time = time()
    history = trainer.train()
    end_time = time()
    
    print(f"Training completed in {(end_time - start_time)/60:.2f} minutes.")
    
    print("Saving model...")
    trainer.base_network.save('../models/triplet_base.keras')
    print("Model saved")
    
    PlotUtility.plot_training_history(
        history_file='../models/triplet_history.json',
        output_path='../plots/triplet_history.png'
    )
    
    print("Finding optimal threshold on validation set...")
    best_treshold, best_acc = trainer.find_optimal_threshold()
    
    trainer.evaluate(
        output_file='plots/triplet_embeddings_separation.png',
    )
    
    
if __name__ == "__main__":
    main()
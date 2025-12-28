from time import time
import pandas as pd

from ensemble.ensemble_trainer import EnsembleTrainer


def main():
    triplet_model_path = '../models/triplet_base.keras'
    samples_dir = '../data/samples'
    data_dir = '../data'
    
    trainer = EnsembleTrainer(
        samples_dir=samples_dir,
        triplet_model_path=triplet_model_path
    )
    
    print("Loading data...")
    
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    val_df = pd.read_csv(f'{data_dir}/validation.csv')
    test_df = pd.read_csv(f'{data_dir}/test.csv')
    
    print(f"Loaded {len(train_df)} training, {len(val_df)} validation and {len(test_df)} test samples.")
    
    start_time = time()
    
    X_train, y_train = trainer.extract_features_from_df(train_df, 'Extracting features from train df...')
    X_val, y_val = trainer.extract_features_from_df(val_df, 'Extracting features from val df...')
    # X_test, _ = trainer.extract_features_from_df(test_df, 'Extracting features from test df...')
    
    print(f"Feature extraction completed in {(time() - start_time)/60:.2f} minutes.")
    
    print("Training ensemble model...")
    trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )
    
    trainer.compute_metrics_on_validation(
        X_val=X_val,
        y_val=y_val,
    )
    
    # print(f"Total training time: {(time() - start_time)/60:.2f} minutes.")
    
    # print("Evaluating ensemble model on test set...")
    # trainer.predict_test(
    #     X_test=X_test,
    #     test_df=test_df,
    #     output_file='ensemble_test_predictions.csv'
    # )
    
    
if __name__ == "__main__":
    main()
    
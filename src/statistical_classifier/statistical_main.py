from time import time
import pandas as pd

from statistical_classifier.statistical_trainer import StatisticalClassifier


def main():
    samples_dir = '../data/samples'
    data_dir = '../data'
    
    classifier = StatisticalClassifier(
        samples_dir=samples_dir
    )
    
    print("Loading data...")
    
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    val_df = pd.read_csv(f'{data_dir}/validation.csv')
    test_df = pd.read_csv(f'{data_dir}/test.csv')
    
    print(f"Loaded {len(train_df)} training, {len(val_df)} validation and {len(test_df)} test samples.")
    
    start_time = time()
    
    print("\nExtracting statistical features...")
    X_train, y_train = classifier.extract_features_from_df(train_df, 'Extracting features from train df...')
    X_val, y_val = classifier.extract_features_from_df(val_df, 'Extracting features from val df...')
    # X_test, _ = classifier.extract_features_from_df(test_df, 'Extracting features from test df...')
    
    print(f"Feature extraction completed in {(time() - start_time)/60:.2f} minutes.")
    print(f"Feature dimensionality: {X_train.shape[1]} features")
    
    print("\nTraining statistical classifiers...")
    classifier.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )
    
    print(f"\nTotal training time: {(time() - start_time)/60:.2f} minutes.")
    
    print("\nComputing detailed metrics on validation set...")
    classifier.compute_metrics_on_validation(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )
    
    # print("\nEvaluating model on test set...")
    # classifier.predict_test(
    #     X_test=X_test,
    #     test_df=test_df,
    #     output_file='statistical_test_predictions.csv'
    # )
    
    print("\nStatistical classifier pipeline completed successfully!")
    
    
if __name__ == "__main__":
    main()
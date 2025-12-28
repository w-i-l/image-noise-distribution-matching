import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy import stats
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb

from utils.plot_utility import PlotUtility


class EnsembleTrainer:
    def __init__(
        self,
        triplet_model_path: str,
        samples_dir: str
    ):
        self.samples_dir = Path(samples_dir)
        self.base_network = keras.models.load_model(
            triplet_model_path, 
            compile=False,
            safe_mode=False
        )
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        
        Path('../plots').mkdir(parents=True, exist_ok=True)
        Path('../models').mkdir(parents=True, exist_ok=True)
        
        print(f"Loaded triplet model from: {triplet_model_path}")
        
        
    ################################################################
    #                     FEATURE EXTRACTION                       #
    ################################################################
    
    def extract_features_from_df(
        self,
        df: pd.DataFrame,
        desc: str = "Extracting features"
    ) -> tuple:
        """
        Extract features for all image pairs in the dataframe.
        
        Returns a tuple (X, y) where X is the feature matrix and y are the labels.
        If 'label' column is not present in df, y will be None.
        """
        
        features_list = []
        labels = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            id_img1 = row['id_noise_1']
            id_img2 = row['id_noise_2']
            
            features = self._extract_all_features(id_img1, id_img2)
            features_list.append(features)
            
            if 'label' in row:
                labels.append(row['label'])
                
        features_df = pd.DataFrame(features_list)
        
        if self.feature_names is None:
            self.feature_names = features_df.columns.tolist()
            
        X = features_df[self.feature_names].values
        y = np.array(labels) if labels else None
        
        return X, y
    
    
    def _extract_statistical_features(
        self,
        img1: np.ndarray,
        img2: np.ndarray
    ) -> dict:
        """
        Extract statistical features from two images.
        
        Returns a dictionary of 16 features.
        """
       
        features = {}
        
        flat1 = img1.flatten()
        flat2 = img2.flatten()
        
        # correlation
        features['pearson'] = np.corrcoef(flat1, flat2)[0, 1]
        features['spearman'] = stats.spearmanr(flat1, flat2)[0]
        
        # pixel-wise differences
        features['mae'] = np.abs(flat1 - flat2).mean()
        features['mse'] = ((flat1 - flat2)**2).mean()
        features['max_diff'] = np.abs(flat1 - flat2).max()
        
        # distribution tests
        features['ks_stat'] = stats.ks_2samp(flat1, flat2)[0]
        
        # moments
        features['mean_diff'] = abs(flat1.mean() - flat2.mean())
        features['std_diff'] = abs(flat1.std() - flat2.std())
        features['skew_diff'] = abs(stats.skew(flat1) - stats.skew(flat2))
        features['kurt_diff'] = abs(stats.kurtosis(flat1) - stats.kurtosis(flat2))
        
        # quadrant correlations
        h, w = img1.shape
        h2, w2 = h // 2, w // 2
        
        features['corr_tl'] = np.corrcoef(
            img1[:h2, :w2].flatten(),
            img2[:h2, :w2].flatten()
        )[0, 1]
        
        features['corr_tr'] = np.corrcoef(
            img1[:h2, w2:].flatten(),
            img2[:h2, w2:].flatten()
        )[0, 1]
        
        features['corr_br'] = np.corrcoef(
            img1[h2:, w2:].flatten(),
            img2[h2:, w2:].flatten()
        )[0, 1]
        
        features['corr_bl'] = np.corrcoef(
            img1[h2:, :w2].flatten(),
            img2[h2:, :w2].flatten()
        )[0, 1]
        
        # histogram similarity
        hist1, _ = np.histogram(flat1, bins=50)
        hist2, _ = np.histogram(flat2, bins=50)
        hist1 = hist1 / (hist1.sum() + 1e-10)
        hist2 = hist2 / (hist2.sum() + 1e-10)
        
        features['hist_chi2'] = np.sum((hist1 - hist2)**2 / (hist1 + hist2 + 1e-10))
        features['hist_intersect'] = np.minimum(hist1, hist2).sum()
        
        # handle NaN or inf values
        for key in features:
            if not np.isfinite(features[key]):
                features[key] = 0.0
        
        return features
    

    def _extract_embeddings(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        no_of_layers: int = 10
    ) -> dict:
        """
        Extract embeddings for two images using the base network.
        
        Returns a dictionary of embedding features.
        """
        
        # normalize images
        img1 = (img1 - img1.mean()) / (img1.std() + 1e-8)
        img2 = (img2 - img2.mean()) / (img2.std() + 1e-8)
        
        emb1 = self.base_network.predict(
            img1[np.newaxis, ..., np.newaxis],
            verbose=0
        )[0]
        
        emb2 = self.base_network.predict(
            img2[np.newaxis, ..., np.newaxis],
            verbose=0
        )[0]
        
        features = {}
        features['embed_dist'] = np.linalg.norm(emb1 - emb2)
        features['embed_cos'] = np.dot(emb1, emb2)  # Already normalized
        features['embed_mae'] = np.abs(emb1 - emb2).mean()
        features['embed_max'] = np.abs(emb1 - emb2).max()
        
        # prevent too many features by taking 
        # only first k dimensions differences
        for i in range(min(no_of_layers, len(emb1))):
            features[f'embed_d{i}'] = abs(emb1[i] - emb2[i])
        
        return features 
    

    def _extract_all_features(
        self,
        id_img1: str,
        id_img2: str
    ) -> dict:
        """
        Extract all features (statistical + embedding) from two images.
        
        Returns a dictionary of features.
        """
        img1_path = self.samples_dir / f"{id_img1}.npy"
        img2_path = self.samples_dir / f"{id_img2}.npy"
        
        img1 = np.load(img1_path).astype(np.float32)
        img2 = np.load(img2_path).astype(np.float32)
        
        features = {}
        
        stat_features = self._extract_statistical_features(img1, img2)
        embed_features = self._extract_embeddings(img1, img2, no_of_layers=10)
        
        features.update(stat_features)
        features.update(embed_features)
        
        return features
    
    
    ################################################################
    #                       TRAINING METHODS                       #
    ################################################################
    
    def train(
        self, 
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """
        Train ensemble model using multiple classifiers and select the best one.
        """
        
        print("Training ensemble classifiers...")
        
        classifiers = {
            'Logistic Regression': self._train_logistic_regression,
            'LightGBM': self._train_lightgbm,
            'Gradient Boosting': self._train_gradient_boosting,
            'Random Forest': self._train_random_forest,
            'XGBoost': self._train_xgboost,
        }
        
        best_acc = 0.0
        best_model = None
        best_name = ""
        
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        for name, train_fn in classifiers.items():
            print(f"Training {name} classifier...")
            
            model, acc = train_fn(X_train, y_train, X_val, y_val)
            
            print(f"{name} accuracy: {acc*100:.2f}%")
            
            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_name = name
                
        self.model = best_model
        print(f"Selected best model: {best_name} with accuracy: {best_acc*100:.2f}%")
        
        self._print_features_importance()
        
        PlotUtility.plot_confusion_matrix(
            predictions=self.model.predict(X_val),
            labels=y_val,
            output_path='../plots/ensemble_confusion_matrix.png'
        )
    
    
    def _train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple:
        lgbm = lgbm = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=8,
            num_leaves=50,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1
        )
        
        lgbm.fit(X_train, y_train)
        lgbm_pred = lgbm.predict(X_val)
        
        lgbm_acc = accuracy_score(y_val, lgbm_pred)
        
        return lgbm, lgbm_acc
    

    def _train_gradient_boosting(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple:
        gb = GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=10,
        )
        
        gb.fit(X_train, y_train)
        gb_pred = gb.predict(X_val)
        
        gb_acc = accuracy_score(y_val, gb_pred)
        
        return gb, gb_acc
    
    
    def _train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple:
        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_val)
        
        rf_acc = accuracy_score(y_val, rf_pred)
        
        return rf, rf_acc
    
    
    def _train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple:
        xgboost = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        xgboost.fit(X_train, y_train)
        xgb_pred = xgboost.predict(X_val)
        
        xgb_acc = accuracy_score(y_val, xgb_pred)
        
        return xgboost, xgb_acc
    
    
    def _train_logistic_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple:
        lr = LogisticRegression(
            max_iter=1000,
            n_jobs=-1
        )
        
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_val)
        
        lr_acc = accuracy_score(y_val, lr_pred)
        
        return lr, lr_acc
    
    
    #################################################################
    #                         EVALUATING                            #
    #################################################################
    
    def predict_test(
        self,
        X_test: np.ndarray,
        test_df: pd.DataFrame,
        output_file='ensemble_test_predictions.csv'
    ) -> None:
        """
        Predict on test set and save predictions to CSV.
        """
        
        if self.model is None:
            print("Model not trained yet.")
            return
        
        X_test = self.scaler.transform(X_test)
        test_preds = self.model.predict(X_test)
        
        proba = self.model.predict_proba(X_test)
        confidience = np.max(proba, axis=1)
        
        print(f"Mean confidence on test set: {confidience.mean():.4f}")
        print(f"Low confidence predictions (<0.6): {(confidience < 0.6).sum()}")
        
        id_pairs = [
            f"{row['id_noise_1']}_{row['id_noise_2']}"
            for _, row in test_df.iterrows()
        ]
        
        submission = pd.DataFrame({
            'id_pair': id_pairs,
            'label': test_preds
        })
        
        submission.to_csv(output_file, index=False)
        print(f"Saved test predictions to: {output_file}")
        
    
    def compute_metrics_on_validation(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> None:
        """
        Compute and print MAE, MSE, Spearman, and Kendall's tau on validation set
        for the best trained classifier.
        """
        
        if self.model is None:
            print("Model not trained yet. Please call train() method first.")
            return
        
        print("\n" + "="*70)
        print("VALIDATION METRICS FOR BEST MODEL")
        print("="*70)
        
        X_val_scaled = self.scaler.transform(X_val)

        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        elif hasattr(self.model, 'decision_function'):
            y_pred_proba = self.model.decision_function(X_val_scaled)
        else:
            y_pred_proba = self.model.predict(X_val_scaled).astype(float)
        
        y_pred = self.model.predict(X_val_scaled)
        
        mae = np.mean(np.abs(y_val - y_pred_proba))
        mse = np.mean((y_val - y_pred_proba) ** 2)
        rmse = np.sqrt(mse)
        
        spearman_corr, spearman_pval = stats.spearmanr(y_val, y_pred_proba)
        
        kendall_tau, kendall_pval = stats.kendalltau(y_val, y_pred_proba)
        
        print(f"\nClassification Metrics:")
        
        print(f"\nRegression-style Metrics (on probabilities):")
        print(f"  MAE:             {mae:.4f}")
        print(f"  MSE:             {mse:.4f}")
        print(f"  RMSE:            {rmse:.4f}")
        
        print(f"\nCorrelation Metrics:")
        print(f"  Spearman ρ:      {spearman_corr:.4f} (p-value: {spearman_pval:.4e})")
        print(f"  Kendall's τ:     {kendall_tau:.4f} (p-value: {kendall_pval:.4e})")
        
        print("\n" + "="*70)
    
    
    #################################################################
    #                          UTILITIES                            #
    #################################################################
    
    def _print_features_importance(self):
        """
        Print feature importance for tree-based models.
        """
        if self.model is None:
            print("Model not trained yet.")
            return
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = sorted(
                zip(self.feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )
            
            print("Feature Importances:")
            for feature, importance in feature_importance:
                print(f"{feature}: {importance:.4f}")
        else:
            print("Feature importance not available for this model type.")
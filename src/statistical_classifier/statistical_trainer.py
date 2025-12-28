import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import cv2

from utils.plot_utility import PlotUtility


class StatisticalClassifier:
    def __init__(
        self,
        samples_dir: str
    ):
        self.samples_dir = Path(samples_dir)
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        
        Path('../plots').mkdir(parents=True, exist_ok=True)
        Path('../models').mkdir(parents=True, exist_ok=True)
        
        print(f"Initialized Statistical Classifier with samples from: {samples_dir}")
        
        
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
        
        Returns a dictionary of 25 features.
        """
       
        features = {}
        
        flat1 = img1.flatten()
        flat2 = img2.flatten()
        
        # correlation (8 features)
        features['pearson'] = np.corrcoef(flat1, flat2)[0, 1]
        features['spearman'] = stats.spearmanr(flat1, flat2)[0]
        
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
        
        # FFT correlation
        fft1 = np.fft.fft2(img1)
        fft2 = np.fft.fft2(img2)
        fft_mag1 = np.abs(fft1).flatten()
        fft_mag2 = np.abs(fft2).flatten()
        features['fft_corr'] = np.corrcoef(fft_mag1, fft_mag2)[0, 1]
        
        # gradient correlation
        grad1_x = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
        grad1_y = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
        grad1_mag = np.sqrt(grad1_x**2 + grad1_y**2)
        
        grad2_x = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
        grad2_y = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)
        grad2_mag = np.sqrt(grad2_x**2 + grad2_y**2)
        
        features['grad_corr'] = np.corrcoef(
            grad1_mag.flatten(),
            grad2_mag.flatten()
        )[0, 1]
        
        # distribution tests (6 features)
        ks_result = stats.ks_2samp(flat1, flat2)
        features['ks_statistic'] = ks_result[0]
        features['ks_pvalue'] = ks_result[1]
        
        features['wasserstein_dist'] = wasserstein_distance(flat1, flat2)
        
        # histogram features
        hist1, _ = np.histogram(flat1, bins=50)
        hist2, _ = np.histogram(flat2, bins=50)
        hist1 = hist1 / (hist1.sum() + 1e-10)
        hist2 = hist2 / (hist2.sum() + 1e-10)
        
        features['hist_chi2'] = np.sum((hist1 - hist2)**2 / (hist1 + hist2 + 1e-10))
        features['hist_intersect'] = np.minimum(hist1, hist2).sum()
        
        # bhattacharyya distance
        features['bhattacharyya'] = -np.log(np.sum(np.sqrt(hist1 * hist2)) + 1e-10)
        
        # moments (5 features)
        features['mean_diff'] = abs(flat1.mean() - flat2.mean())
        features['std_diff'] = abs(flat1.std() - flat2.std())
        features['var_diff'] = abs(flat1.var() - flat2.var())
        features['skew_diff'] = abs(stats.skew(flat1) - stats.skew(flat2))
        features['kurt_diff'] = abs(stats.kurtosis(flat1) - stats.kurtosis(flat2))
        
        # pixel-wise differences (4 features)
        features['mae'] = np.abs(flat1 - flat2).mean()
        features['mse'] = ((flat1 - flat2)**2).mean()
        features['max_diff'] = np.abs(flat1 - flat2).max()
        features['median_diff'] = np.median(np.abs(flat1 - flat2))
        
        # frequency domain (2 features)
        features['fft_mae'] = np.abs(fft_mag1 - fft_mag2).mean()
        
        # handle NaN or inf values
        for key in features:
            if not np.isfinite(features[key]):
                features[key] = 0.0
        
        return features
    

    def _extract_all_features(
        self,
        id_img1: str,
        id_img2: str
    ) -> dict:
        """
        Extract all statistical features from two images.
        
        Returns a dictionary of 25 features.
        """
        img1_path = self.samples_dir / f"{id_img1}.npy"
        img2_path = self.samples_dir / f"{id_img2}.npy"
        
        img1 = np.load(img1_path).astype(np.float32)
        img2 = np.load(img2_path).astype(np.float32)
        
        features = self._extract_statistical_features(img1, img2)
        
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
        Train statistical classifiers and select the best one.
        """
        
        print("Training statistical classifiers...")
        
        classifiers = {
            'Logistic Regression': self._train_logistic_regression,
            'Random Forest': self._train_random_forest,
            'Gradient Boosting': self._train_gradient_boosting,
            'Ensemble': self._train_ensemble,
        }
        
        best_acc = 0.0
        best_model = None
        best_name = ""
        
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        results = {}
        
        for name, train_fn in classifiers.items():
            print(f"\nTraining {name} classifier...")
            
            model, metrics = train_fn(X_train, y_train, X_val, y_val)
            
            results[name] = metrics
            
            print(f"{name} - Accuracy: {metrics['accuracy']:.4f}, "
                  f"Precision: {metrics['precision']:.4f}, "
                  f"Recall: {metrics['recall']:.4f}, "
                  f"F1-Score: {metrics['f1']:.4f}")
            
            if metrics['accuracy'] > best_acc:
                best_acc = metrics['accuracy']
                best_model = model
                best_name = name
                
        self.model = best_model
        print(f"\nSelected best model: {best_name} with accuracy: {best_acc:.4f}")
        
        self._print_results_table(results)
        self._print_features_importance()
        
        PlotUtility.plot_confusion_matrix(
            predictions=self.model.predict(X_val),
            labels=y_val,
            output_path='../plots/statistical_confusion_matrix.png'
        )
        
        self._plot_best_model_decision_boundary(
            X_val,
            y_val,
            output_path='../plots/statistical_decision_boundary.png'
        )
    
    
    def _train_logistic_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple:
        lr = LogisticRegression(
            C=1.0,
            max_iter=1000,
            n_jobs=-1
        )
        
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_val)
        
        metrics = {
            'accuracy': accuracy_score(y_val, lr_pred),
            'precision': precision_score(y_val, lr_pred),
            'recall': recall_score(y_val, lr_pred),
            'f1': f1_score(y_val, lr_pred)
        }
        
        return lr, metrics
    

    def _train_gradient_boosting(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple:
        gb = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
        )
        
        gb.fit(X_train, y_train)
        gb_pred = gb.predict(X_val)
        
        metrics = {
            'accuracy': accuracy_score(y_val, gb_pred),
            'precision': precision_score(y_val, gb_pred),
            'recall': recall_score(y_val, gb_pred),
            'f1': f1_score(y_val, gb_pred)
        }
        
        return gb, metrics
    
    
    def _train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple:
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_val)
        
        metrics = {
            'accuracy': accuracy_score(y_val, rf_pred),
            'precision': precision_score(y_val, rf_pred),
            'recall': recall_score(y_val, rf_pred),
            'f1': f1_score(y_val, rf_pred)
        }
        
        return rf, metrics
    
    
    def _train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple:
        lr = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1)
        rf = RandomForestClassifier(n_estimators=300, max_depth=15, n_jobs=-1)
        gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5)
        
        ensemble = VotingClassifier(
            estimators=[
                ('lr', lr),
                ('rf', rf),
                ('gb', gb)
            ],
            voting='soft'
        )
        
        ensemble.fit(X_train, y_train)
        ensemble_pred = ensemble.predict(X_val)
        
        metrics = {
            'accuracy': accuracy_score(y_val, ensemble_pred),
            'precision': precision_score(y_val, ensemble_pred),
            'recall': recall_score(y_val, ensemble_pred),
            'f1': f1_score(y_val, ensemble_pred)
        }
        
        return ensemble, metrics
    
    
    #################################################################
    #                         EVALUATING                            #
    #################################################################
    
    def predict_test(
        self,
        X_test: np.ndarray,
        test_df: pd.DataFrame,
        output_file='statistical_test_predictions.csv'
    ) -> None:
        
        if self.model is None:
            print("Model not trained yet.")
            return
        
        X_test = self.scaler.transform(X_test)
        test_preds = self.model.predict(X_test)
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_test)
            confidence = np.max(proba, axis=1)
            
            print(f"Mean confidence on test set: {confidence.mean():.4f}")
            print(f"Low confidence predictions (<0.6): {(confidence < 0.6).sum()}")
        
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
        y_val: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> None:
        """
        Compute and print accuracy, precision, recall, and F1-score on validation set
        for each trained classifier.
        """
        
        print("\n" + "="*70)
        print("VALIDATION METRICS FOR ALL CLASSIFIERS")
        print("="*70)
        
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Define all classifiers
        classifiers = {
            'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1),
            'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=15, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5),
        }
        
        # Train and evaluate each classifier
        for name, clf in classifiers.items():
            print(f"\n{name}:")
            print("-" * 70)
            
            # Train the classifier
            clf.fit(X_train_scaled, y_train)
            
            # Get predictions
            y_pred = clf.predict(X_val_scaled)
            
            # Compute metrics
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            
            mse = np.mean((y_val - y_pred) ** 2)
            mae = np.mean(np.abs(y_val - y_pred))
            spearman_corr, _ = stats.spearmanr(y_val, y_pred)
            kendall_tau, _ = stats.kendalltau(y_val, y_pred)
            
            # Print results
            print(f"  Accuracy:        {accuracy:.4f}")
            print(f"  Precision:       {precision:.4f}")
            print(f"  Recall:          {recall:.4f}")
            print(f"  F1-Score:        {f1:.4f}")
            print(f"  MAE:             {mae:.4f}")
            print(f"  MSE:             {mse:.4f}")
            print(f"  Spearman Corr.:  {spearman_corr:.4f}")
            print(f"  Kendall's Tau:   {kendall_tau:.4f}")
        
        # Add ensemble
        print(f"\nEnsemble:")
        print("-" * 70)
        
        lr = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1)
        rf = RandomForestClassifier(n_estimators=300, max_depth=15, n_jobs=-1)
        gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5)
        
        ensemble = VotingClassifier(
            estimators=[('lr', lr), ('rf', rf), ('gb', gb)],
            voting='soft'
        )
        
        ensemble.fit(X_train_scaled, y_train)
        y_pred = ensemble.predict(X_val_scaled)
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        mae = np.mean(np.abs(y_val - y_pred))
        mse = np.mean((y_val - y_pred) ** 2)
        spearman_corr, _ = stats.spearmanr(y_val, y_pred)
        kendall_tau, _ = stats.kendalltau(y_val, y_pred)
        
        print(f"  Accuracy:        {accuracy:.4f}")
        print(f"  Precision:       {precision:.4f}")
        print(f"  Recall:          {recall:.4f}")
        print(f"  F1-Score:        {f1:.4f}")
        print(f"  MAE:             {mae:.4f}")
        print(f"  MSE:             {mse:.4f}")
        print(f"  Spearman Corr.:  {spearman_corr:.4f}")
        print(f"  Kendall's Tau:   {kendall_tau:.4f}")
        
        print("\n" + "="*70)
    
    
    #################################################################
    #                          UTILITIES                            #
    #################################################################
    
    def _print_results_table(self, results: dict):
        print("\n" + "="*70)
        print("Model Performance on Validation Set")
        print("="*70)
        print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 70)
        
        for name, metrics in results.items():
            print(f"{name:<25} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
        
        print("="*70 + "\n")
    
    
    def _print_features_importance(self):
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
            
            print("\n" + "="*70)
            print("Top 20 Feature Importances")
            print("="*70)
            print(f"{'Rank':<6} {'Feature Name':<25} {'Importance':<12}")
            print("-" * 70)
            
            for rank, (feature, importance) in enumerate(feature_importance[:20], 1):
                print(f"{rank:<6} {feature:<25} {importance:<12.4f}")
            
            print("="*70 + "\n")
        elif hasattr(self.model, 'named_estimators_'):
            # For ensemble, use gradient boosting importance
            if 'gb' in self.model.named_estimators_:
                importances = self.model.named_estimators_['gb'].feature_importances_
                feature_importance = sorted(
                    zip(self.feature_names, importances),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                print("\n" + "="*70)
                print("Top 20 Feature Importances (from Gradient Boosting in Ensemble)")
                print("="*70)
                print(f"{'Rank':<6} {'Feature Name':<25} {'Importance':<12}")
                print("-" * 70)
                
                for rank, (feature, importance) in enumerate(feature_importance[:20], 1):
                    print(f"{rank:<6} {feature:<25} {importance:<12.4f}")
                
                print("="*70 + "\n")
        else:
            print("Feature importance not available for this model type.")
            
    
    def _plot_best_model_decision_boundary(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        output_path: str = '../plots/statistical_decision_boundary.png'
    ):
        """
        Plot decision boundary of the best model using prediction probabilities.
        Finds optimal threshold and plots the distribution of predictions.
        """
        
        if self.model is None:
            print("Model not trained yet.")
            return
        
        if not hasattr(self.model, 'predict_proba'):
            print("Model does not support probability predictions.")
            return
        
        y_proba = self.model.predict_proba(X_val)[:, 1]
        
        thresholds = np.linspace(0, 1, 100)
        best_acc = 0.0
        best_thresh = 0.5
        
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            acc = accuracy_score(y_val, y_pred)
            
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        
        print(f"Best threshold: {best_thresh:.3f} with accuracy: {best_acc:.4f}")
        
        same_distributions = y_proba[y_val == 1].tolist()
        diff_distributions = y_proba[y_val == 0].tolist()
        
        PlotUtility.plot_embedings_separation(
            same_distributions=same_distributions,
            diff_distributions=diff_distributions,
            best_thresh=best_thresh,
            output_path=output_path
        )
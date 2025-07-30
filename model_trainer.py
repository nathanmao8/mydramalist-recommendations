# model_trainer.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score
from typing import Dict, List, Tuple
import time

class ModelTrainer:
    def __init__(self):
        # Traditional models
        self.rf_traditional = RandomForestRegressor(
            n_estimators=100, 
            max_features='sqrt',
            random_state=42
        )
        self.svm_traditional = SVR(kernel='rbf', gamma='scale')
        
        # BERT-enhanced models
        self.rf_bert = RandomForestRegressor(
            n_estimators=100, 
            max_features='sqrt',
            random_state=42
        )
        self.svm_bert = SVR(kernel='rbf', gamma='scale')
        
        self.threshold = 7.5  # Threshold for high ratings (7.5+)
    
    def train_models(self, X_traditional: np.ndarray, X_hybrid: np.ndarray, y: np.ndarray) -> Tuple[Dict, Dict, Dict]:
        """Train both traditional and hybrid models with LOOCV and return LOOCV predictions."""
        
        print("Training traditional models...")
        traditional_results, rf_trad_preds, svm_trad_preds, true_values = self.train_model_set(
            X_traditional, y, 
            self.rf_traditional, self.svm_traditional,
            model_prefix="Traditional"
        )
        
        print("Training BERT-enhanced models...")
        bert_results, rf_bert_preds, svm_bert_preds, _ = self.train_model_set(
            X_hybrid, y,
            self.rf_bert, self.svm_bert,
            model_prefix="BERT"
        )
        
        # Combine results
        all_results = {**traditional_results, **bert_results}
        
        print("Capturing feature importances from Random Forest models...")
        feature_importances = {
            'rf_traditional': self.rf_traditional.feature_importances_,
            'rf_bert': self.rf_bert.feature_importances_
        }
        
        # Add the captured importances to the results dictionary
        all_results['feature_importances'] = feature_importances

        # Create model dictionary for predictions
        trained_models = {
            'rf_traditional': self.rf_traditional,
            'svm_traditional': self.svm_traditional,
            'rf_bert': self.rf_bert,
            'svm_bert': self.svm_bert
        }
        
        # Create a dictionary to hold all LOOCV predictions
        loocv_predictions = {
            'true_ratings': true_values,
            'rf_traditional_preds': rf_trad_preds,
            'svm_traditional_preds': svm_trad_preds,
            'rf_bert_preds': rf_bert_preds,
            'svm_bert_preds': svm_bert_preds
        }

        return trained_models, all_results, loocv_predictions
    
    def train_model_set(self, X: np.ndarray, y: np.ndarray, rf_model, svm_model, model_prefix: str) -> Tuple[Dict, list, list, list]:
        """Train a set of models using LOOCV, return evaluation results and predictions."""
        
        # Leave-One-Out Cross-Validation
        loo = LeaveOneOut()
        
        rf_predictions = []
        svm_predictions = []
        true_values = []
        
        start_time = time.time()
        
        print(f"Performing LOOCV for {model_prefix} models...")
        
        for i, (train_idx, test_idx) in enumerate(loo.split(X)):
            if i % 10 == 0:
                print(f"  Processing fold {i+1}/{len(X)}")
            
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]
            
            # Bootstrap within fold for stability
            bootstrap_rf_preds = []
            bootstrap_svm_preds = []
            
            for _ in range(100):  # Bootstrap iterations
                # Bootstrap sample
                n_samples = len(X_train_fold)
                bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
                X_bootstrap = X_train_fold[bootstrap_idx]
                y_bootstrap = y_train_fold[bootstrap_idx]
                
                # Train models on bootstrap sample
                rf_temp = RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=42)
                svm_temp = SVR(kernel='rbf', gamma='scale')
                
                rf_temp.fit(X_bootstrap, y_bootstrap)
                svm_temp.fit(X_bootstrap, y_bootstrap)
                
                # Predict on test sample
                bootstrap_rf_preds.append(rf_temp.predict(X_test_fold)[0])
                bootstrap_svm_preds.append(svm_temp.predict(X_test_fold)[0])
            
            # Average bootstrap predictions
            rf_predictions.append(np.mean(bootstrap_rf_preds))
            svm_predictions.append(np.mean(bootstrap_svm_preds))
            true_values.append(y_test_fold[0])
        
        training_time = time.time() - start_time
        
        # Calculate evaluation metrics
        results = self.calculate_evaluation_metrics(
            true_values, rf_predictions, svm_predictions, model_prefix, training_time
        )
        
        # Train final models on full dataset
        print(f"Training final {model_prefix} models on full dataset...")
        rf_model.fit(X, y)
        svm_model.fit(X, y)
        
        return results, rf_predictions, svm_predictions, true_values
    
    def calculate_evaluation_metrics(self, true_values: List, rf_preds: List, svm_preds: List, 
                                   model_prefix: str, training_time: float) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        
        def calculate_metrics(y_true, y_pred, model_name):
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # 1. Convert the regression problem to a binary classification problem
            y_true_binary = (y_true >= self.threshold).astype(int)
            y_pred_binary = (y_pred >= self.threshold).astype(int)
            
            # 2. Calculate precision and recall using sklearn
            #    `zero_division=0` prevents warnings if no positive predictions are made.
            precision_high = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall_high = recall_score(y_true_binary, y_pred_binary, zero_division=0)

            # --- MAE for High-Rated Items ---
            
            # 1. Create a boolean mask to select only items the user actually rated highly
            high_rated_mask = (y_true >= self.threshold)
            
            # 2. Apply the mask to both the true and predicted ratings
            y_true_high_rated = y_true[high_rated_mask]
            y_pred_high_rated = y_pred[high_rated_mask]
            
            # 3. Calculate MAE on this filtered subset
            #    Handle the case where there are no high-rated items in the set.
            if len(y_true_high_rated) > 0:
                mae_high = mean_absolute_error(y_true_high_rated, y_pred_high_rated)
            else:
                mae_high = 0.0

            # # Precision for high ratings (7.5+)
            # high_true_indices = [i for i, val in enumerate(y_true) if val >= self.threshold]
            # high_pred_indices = [i for i, val in enumerate(y_pred) if val >= self.threshold]
            
            # if len(high_pred_indices) > 0:
            #     precision_high = len(set(high_true_indices) & set(high_pred_indices)) / len(high_pred_indices)
            # else:
            #     precision_high = 0.0
            
            return {
                f'{model_name}_RMSE': rmse,
                f'{model_name}_MAE': mae,
                f'{model_name}_R2': r2,
                f'{model_name}_Precision_7.5+': precision_high,
                f'{model_name}_Recall_7.5+': recall_high,
                f'{model_name}_MAE_High_Rated': mae_high
            }
        
        # Convert lists to NumPy arrays for easier manipulation
        true_values_np = np.array(true_values)
        rf_preds_np = np.array(rf_preds)
        svm_preds_np = np.array(svm_preds)

        # Calculate metrics for both models
        rf_metrics = calculate_metrics(true_values_np, rf_preds_np, f'RF_{model_prefix}')
        svm_metrics = calculate_metrics(true_values_np, svm_preds_np, f'SVM_{model_prefix}')

        # Add training time
        rf_metrics[f'RF_{model_prefix}_TrainingTime'] = training_time / 2  # Approximate split
        svm_metrics[f'SVM_{model_prefix}_TrainingTime'] = training_time / 2
        
        return {**rf_metrics, **svm_metrics}

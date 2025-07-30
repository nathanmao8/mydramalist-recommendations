# evaluator.py
from typing import Dict, List
import pandas as pd
from sklearn.inspection import permutation_importance
import numpy as np

class Evaluator:
    def display_results(self, evaluation_results: Dict):
        """Display comprehensive evaluation results for all models."""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL EVALUATION RESULTS")
        print("="*80)
        
        # Create comparison table
        self.create_comparison_table(evaluation_results)
        
        # Feature importance analysis
        self.display_feature_analysis()
        
        # Model recommendations
        self.display_recommendations(evaluation_results)
    
    def save_loocv_predictions(self, dramas: List[Dict], loocv_predictions: Dict, filename: str = 'loocv_predictions_watched.csv'):
        """
        Creates and saves a DataFrame of LOOCV predictions for watched dramas.
        
        Args:
            dramas (List[Dict]): The original list of watched drama data.
            loocv_predictions (Dict): The dictionary containing true ratings and model predictions.
            filename (str): The name of the output CSV file.
        """
        print(f"\nSaving LOOCV predictions for watched dramas to {filename}...")

        # Ensure we have the same number of dramas and predictions
        if len(dramas) != len(loocv_predictions['true_ratings']):
            print("Error: Mismatch between number of dramas and number of LOOCV predictions.")
            return

        # Create a DataFrame
        df_data = {
            'Drama_Title': [d.get('title', 'Unknown') for d in dramas],
            'Actual_Rating': loocv_predictions['true_ratings'],
            'RF_Traditional_Pred': loocv_predictions['rf_traditional_preds'],
            'SVM_Traditional_Pred': loocv_predictions['svm_traditional_preds'],
            'RF_BERT_Pred': loocv_predictions['rf_bert_preds'],
            'SVM_BERT_Pred': loocv_predictions['svm_bert_preds']
        }
        
        df = pd.DataFrame(df_data)

        # Calculate prediction errors
        df['RF_Trad_Error'] = df['RF_Traditional_Pred'] - df['Actual_Rating']
        df['SVM_Trad_Error'] = df['SVM_Traditional_Pred'] - df['Actual_Rating']
        df['RF_BERT_Error'] = df['RF_BERT_Pred'] - df['Actual_Rating']
        df['SVM_BERT_Error'] = df['SVM_BERT_Pred'] - df['Actual_Rating']

        # Round the float columns for better readability
        float_cols = [col for col in df.columns if 'Pred' in col or 'Error' in col or 'Rating' in col]
        df[float_cols] = df[float_cols].round(2)

        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"Successfully saved {len(df)} predictions.")

        # Display the dramas with the highest and lowest predicted ratings from the best model
        # For this example, let's assume RF_BERT is a good model to inspect
        if 'RF_BERT_Pred' in df.columns:
            print("\n--- LOOCV Prediction Highlights (RF_BERT Model) ---")
            print("\nHighest Predictions:")
            print(df.nlargest(5, 'RF_BERT_Pred')[['Drama_Title', 'Actual_Rating', 'RF_BERT_Pred']])
            print("\nLowest Predictions:")
            print(df.nsmallest(5, 'RF_BERT_Pred')[['Drama_Title', 'Actual_Rating', 'RF_BERT_Pred']])
            print("\nLargest Over-predictions (Model was too optimistic):")
            print(df.nlargest(5, 'RF_BERT_Error')[['Drama_Title', 'Actual_Rating', 'RF_BERT_Pred', 'RF_BERT_Error']])
            print("\nLargest Under-predictions (Model was too pessimistic):")
            print(df.nsmallest(5, 'RF_BERT_Error')[['Drama_Title', 'Actual_Rating', 'RF_BERT_Pred', 'RF_BERT_Error']])

    def create_comparison_table(self, results: Dict):
        """Create and display model comparison table."""
        
        print("\nModel Performance Comparison:")
        print("-" * 80)
        print(f"{'Model':<25} {'RMSE':<8} {'MAE':<8} {'R²':<8} {'Precision@7.5+':<15} {'Time(s)':<10}")
        print("-" * 80)
        
        models = [
            ('RF_Traditional', 'Random Forest (Trad)'),
            ('SVM_Traditional', 'SVM (Traditional)'),
            ('RF_BERT', 'Random Forest (BERT)'),
            ('SVM_BERT', 'SVM (BERT)')
        ]
        
        for model_key, model_name in models:
            rmse = results.get(f'{model_key}_RMSE', 0)
            mae = results.get(f'{model_key}_MAE', 0)
            r2 = results.get(f'{model_key}_R2', 0)
            precision = results.get(f'{model_key}_Precision_7.5+', 0)
            time_taken = results.get(f'{model_key}_TrainingTime', 0)
            
            print(f"{model_name:<25} {rmse:<8.3f} {mae:<8.3f} {r2:<8.3f} {precision:<15.3f} {time_taken:<10.1f}")
    
    def display_feature_analysis(self):
        """Display feature importance analysis."""
        
        print("\n" + "="*80)
        print("FEATURE ANALYSIS")
        print("="*80)
        
        print("\nFeature Types:")
        print("• Traditional Features: TF-IDF + Categorical + Sentiment")
        print("• BERT Features: 768-dimensional contextual embeddings")
        print("• Hybrid Features: Traditional + BERT combined")
        
        print("\nKey Insights:")
        print("• Traditional models excel at explicit preferences (cast, genre)")
        print("• BERT models capture nuanced textual patterns")
        print("• Hybrid approach leverages both strengths")
    
    def display_recommendations(self, results: Dict):
        """Display model recommendations based on performance."""
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        # Find best performing models
        best_precision = 0
        best_precision_model = ""
        best_r2 = 0
        best_r2_model = ""
        
        models = ['RF_Traditional', 'SVM_Traditional', 'RF_BERT', 'SVM_BERT']
        
        for model in models:
            precision = results.get(f'{model}_Precision_7.5+', 0)
            r2 = results.get(f'{model}_R2', 0)
            
            if precision > best_precision:
                best_precision = precision
                best_precision_model = model
            
            if r2 > best_r2:
                best_r2 = r2
                best_r2_model = model
        
        print(f"• Best Precision@7.5+: {best_precision_model} ({best_precision:.3f})")
        print(f"• Best R² Score: {best_r2_model} ({best_r2:.3f})")

    def display_feature_importances(self, results: Dict, feature_names: Dict, top_n: int = 20):
        """Displays the top N most important features for Random Forest models."""
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE ANALYSIS (from Random Forest)")
        print("="*80)

        importances_data = results.get('feature_importances', {})
        if not importances_data:
            print("Feature importances not found in results.")
            return

        for model_name, importances in importances_data.items():
            print(f"\n--- Top {top_n} Features for {model_name} ---")
            
            # Determine which set of feature names to use
            if 'traditional' in model_name:
                names = feature_names['traditional']
            else:
                names = feature_names['hybrid']

            # Validate that names and importances have the same length
            if len(names) != len(importances):
                print(f"Warning: Feature names ({len(names)}) and importances ({len(importances)}) have different lengths for {model_name}")
                print("Skipping feature importance display for this model.")
                continue

            # Create a DataFrame for easier sorting and viewing
            df_importances = pd.DataFrame({
                'feature': names,
                'importance': importances
            }).sort_values(by='importance', ascending=False)

            print(df_importances.head(top_n))
    
    def display_permutation_importance(self, models: Dict, X: Dict, y: np.ndarray, feature_names: Dict, top_n: int = 15):
        """Calculates and displays permutation importance for all models."""
        print("\n" + "="*80)
        print("PERMUTATION IMPORTANCE ANALYSIS (Model-Agnostic)")
        print("="*80)

        for model_name, model in models.items():
            print(f"\n--- Top {top_n} Permutation Features for {model_name} ---")

            if 'traditional' in model_name:
                X_data = X['traditional']
                names = feature_names['traditional']
            else:
                X_data = X['hybrid']
                names = feature_names['hybrid']

            # Validate that names and features have the same length
            if len(names) != X_data.shape[1]:
                print(f"Warning: Feature names ({len(names)}) and features ({X_data.shape[1]}) have different lengths for {model_name}")
                print("Skipping permutation importance for this model.")
                continue

            # Calculate permutation importance
            result = permutation_importance(
                model, X_data, y, n_repeats=10, random_state=42, n_jobs=-1
            )

            # Create a DataFrame for display
            df_perm = pd.DataFrame({
                'feature': names,
                'importance_mean': result.importances_mean,
                'importance_std': result.importances_std
            }).sort_values(by='importance_mean', ascending=False)

            print(df_perm.head(top_n))

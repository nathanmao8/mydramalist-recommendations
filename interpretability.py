import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

class ShapExplainer:
    """
    A class to handle model interpretability using the SHAP library.
    It focuses on explaining tree-based models like Random Forest.
    """
    def __init__(self, models: Dict, X_data: Dict, feature_names: Dict):
        """
        Initializes the ShapExplainer with trained models and data.

        Args:
            models (Dict): A dictionary of trained model objects.
            X_data (Dict): A dictionary containing 'traditional' and 'hybrid' feature DataFrames.
            feature_names (Dict): A dictionary containing 'traditional' and 'hybrid' feature name lists.
        """
        self.models = models
        self.X_data = X_data
        self.feature_names = feature_names
        self.explainers = {}
        self.shap_values = {}

        print("Initializing SHAP explainers for tree-based models...")
        self._initialize_explainers()

    def _initialize_explainers(self):
        """Initializes SHAP TreeExplainers for Random Forest models."""
        for model_name, model in self.models.items():
            # SHAP's TreeExplainer is highly optimized for Random Forest
            if 'rf' in model_name.lower():
                print(f"Creating TreeExplainer for: {model_name}")
                self.explainers[model_name] = shap.TreeExplainer(model)

    def explain_models(self):
        """
        Generates SHAP values for all explainable models and creates visualizations.
        """
        print("\n" + "="*80)
        print("SHAP (SHapley Additive exPlanations) ANALYSIS")
        print("This analysis explains how each feature contributes to model predictions.")
        print("="*80)

        for model_name, explainer in self.explainers.items():
            print(f"\n--- Generating SHAP values for: {model_name.upper()} ---")
            
            # Select the correct feature set for the model
            if 'traditional' in model_name:
                X_set = self.X_data['traditional']
                names = self.feature_names['traditional']
            else:
                X_set = self.X_data['hybrid']
                names = self.feature_names['hybrid']

            # Calculate SHAP values for the dataset
            self.shap_values[model_name] = explainer.shap_values(X_set)
            
            # Generate and save the global summary plot
            self.plot_summary(model_name, X_set, names)

            # Generate and save a local force plot for a sample prediction
            # We'll explain the prediction for the first drama in the dataset.
            self.plot_local_explanation(model_name, instance_index=0, feature_names=names)

    def plot_summary(self, model_name: str, X_set: pd.DataFrame, feature_names: List[str]):
        """
        Creates and saves a SHAP summary plot for global feature importance.

        Args:
            model_name (str): The name of the model being explained.
            X_set (pd.DataFrame): The feature matrix used for explanation.
            feature_names (List[str]): The list of feature names.
        """
        print(f"Generating SHAP summary plot for {model_name}...")
        shap_vals = self.shap_values[model_name]

        shap.summary_plot(shap_vals, X_set, feature_names=feature_names, show=False)
        
        plt.title(f'SHAP Global Feature Importance for {model_name}')
        plt.tight_layout()
        
        filename = f'shap_summary_{model_name}.png'
        plt.savefig(filename)
        plt.close()
        print(f"Saved SHAP summary plot to {filename}")

    def plot_local_explanation(self, model_name: str, instance_index: int, feature_names: List[str]):
        """
        Creates and saves a SHAP force plot for a single prediction explanation.

        Args:
            model_name (str): The name of the model being explained.
            instance_index (int): The index of the instance (drama) to explain.
            feature_names (List[str]): The list of feature names.
        """
        print(f"Generating SHAP force plot for instance {instance_index} of {model_name}...")
        
        explainer = self.explainers[model_name]
        shap_vals = self.shap_values[model_name]

        force_plot = shap.force_plot(
            explainer.expected_value, 
            shap_vals[instance_index, :], 
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        
        filename = f'shap_force_plot_{model_name}_instance_{instance_index}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Saved SHAP force plot to {filename}")

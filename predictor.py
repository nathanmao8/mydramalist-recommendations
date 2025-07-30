# predictor.py
import numpy as np
import pandas as pd
from typing import Dict, List

class Predictor:
    def __init__(self):
        self.n_bootstrap = 1000
        self.confidence_level = 0.95
    
    def predict_all_dramas(self, dramas: List[Dict], trained_models: Dict, 
                          X_traditional: np.ndarray, X_hybrid: np.ndarray) -> pd.DataFrame:
        """Generate predictions for all unwatched dramas with both model types."""
        
        predictions = []
        
        for i, drama in enumerate(dramas):
            # Get features for this drama
            traditional_features = X_traditional[i:i+1]
            hybrid_features = X_hybrid[i:i+1]
            
            # Generate predictions from all models
            rf_trad_pred = trained_models['rf_traditional'].predict(traditional_features)[0]
            svm_trad_pred = trained_models['svm_traditional'].predict(traditional_features)[0]
            rf_bert_pred = trained_models['rf_bert'].predict(hybrid_features)[0]
            svm_bert_pred = trained_models['svm_bert'].predict(hybrid_features)[0]
            
            # Calculate ensemble predictions
            traditional_ensemble = (rf_trad_pred + svm_trad_pred) / 2
            bert_ensemble = (rf_bert_pred + svm_bert_pred) / 2
            overall_ensemble = (traditional_ensemble + bert_ensemble) / 2
            
            # Calculate confidence scores
            traditional_confidence = self.calculate_confidence_score([rf_trad_pred, svm_trad_pred])
            bert_confidence = self.calculate_confidence_score([rf_bert_pred, svm_bert_pred])
            overall_confidence = (traditional_confidence + bert_confidence) / 2
            
            predictions.append({
                'Drama_Title': drama.get('title', ''),
                'Drama_ID': drama.get('slug', ''),
                'RF_Traditional': round(rf_trad_pred, 2),
                'SVM_Traditional': round(svm_trad_pred, 2),
                'RF_BERT': round(rf_bert_pred, 2),
                'SVM_BERT': round(svm_bert_pred, 2),
                'Traditional_Ensemble': round(traditional_ensemble, 2),
                'BERT_Ensemble': round(bert_ensemble, 2),
                'Final_Prediction': round(overall_ensemble, 2),
                'Confidence_Score': round(overall_confidence, 3)
            })
        
        return pd.DataFrame(predictions)
    
    def calculate_confidence_score(self, predictions: List[float]) -> float:
        """Calculate confidence score based on prediction variance."""
        
        if len(predictions) <= 1:
            return 0.5
        
        # Calculate variance of predictions
        variance = np.var(predictions)
        
        # Lower variance = higher confidence
        confidence_score = 1 / (1 + variance)
        
        return min(confidence_score, 1.0)

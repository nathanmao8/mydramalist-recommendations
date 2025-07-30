#!/usr/bin/env python3
"""
LOOCV (Leave-One-Out Cross Validation) Evaluator for Taste Analysis
Tests taste profile prediction accuracy by leaving out one drama at a time
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from taste_analyzer import TasteAnalyzer
from semantic_similarity import SemanticSimilarityExtractor
import time
from datetime import datetime

class TasteLOOCVEvaluator:
    def __init__(self):
        self.semantic_extractor = SemanticSimilarityExtractor()
        self.taste_analyzer = TasteAnalyzer(self.semantic_extractor)
        self.results = []
        self.fold_details = []
        
    def run_loocv(self, watched_dramas: List[Dict], ratings: List[float]) -> Dict:
        """
        Run LOOCV evaluation on taste analysis system.
        
        Args:
            watched_dramas: List of drama dictionaries
            ratings: List of corresponding ratings
            
        Returns:
            Dictionary with LOOCV results and metrics
        """
        print("🎯 LOOCV TASTE ANALYSIS EVALUATION")
        print("=" * 50)
        print(f"Testing {len(watched_dramas)} dramas with LOOCV...")
        
        start_time = time.time()
        
        # Store original data
        self.original_dramas = watched_dramas.copy()
        self.original_ratings = ratings.copy()
        
        # Run LOOCV
        for i in range(len(watched_dramas)):
            print(f"\n🔄 Fold {i+1}/{len(watched_dramas)}: Testing '{watched_dramas[i]['title']}'")
            
            # Create training and test sets
            train_dramas = watched_dramas[:i] + watched_dramas[i+1:]
            train_ratings = ratings[:i] + ratings[i+1:]
            test_drama = watched_dramas[i]
            test_rating = ratings[i]
            
            # Build taste profile on training data
            try:
                self.taste_analyzer.analyze_user_taste(train_dramas, train_ratings)
                
                # Calculate similarity for test drama
                test_similarity = self.taste_analyzer.calculate_taste_similarity(test_drama)
                
                # Extract single values from nested dictionaries
                categorical_avg = np.mean(list(test_similarity['categorical'].values()))
                text_avg = np.mean(list(test_similarity['text'].values()))
                
                # Store results
                fold_result = {
                    'fold': i + 1,
                    'test_drama': test_drama['title'],
                    'test_rating': test_rating,
                    'overall_similarity': test_similarity['overall'],
                    'categorical_similarity': categorical_avg,
                    'text_similarity': text_avg,
                    'semantic_similarity': test_similarity['semantic'],
                    'training_size': len(train_dramas),
                    'training_avg_rating': np.mean(train_ratings),
                    'training_std_rating': np.std(train_ratings)
                }
                
                self.fold_details.append(fold_result)
                
                print(f"   Actual Rating: {test_rating}/10")
                print(f"   Overall Similarity: {test_similarity['overall']:.3f}")
                print(f"   Components: Categorical {categorical_avg:.3f}, "
                      f"Text {text_avg:.3f}, Semantic {test_similarity['semantic']:.3f}")
                
            except Exception as e:
                print(f"   ❌ Error in fold {i+1}: {str(e)}")
                fold_result = {
                    'fold': i + 1,
                    'test_drama': test_drama['title'],
                    'test_rating': test_rating,
                    'error': str(e)
                }
                self.fold_details.append(fold_result)
        
        # Calculate overall metrics
        # Dynamic normalization for semantic similarity
        raw_semantic_similarities = [f['semantic_similarity'] for f in self.fold_details if 'semantic_similarity' in f]
        if raw_semantic_similarities:
            min_sem = min(raw_semantic_similarities)
            max_sem = max(raw_semantic_similarities)
            for i, row in enumerate(self.fold_details):
                if 'semantic_similarity' in row:
                    raw_val = row['semantic_similarity']
                    if max_sem > min_sem:
                        norm_val = (raw_val - min_sem) / (max_sem - min_sem)
                    else:
                        norm_val = 0.0
                    self.fold_details[i]['semantic_similarity'] = norm_val
        self.calculate_metrics()
        
        # Generate report
        report = self.generate_report()
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  LOOCV completed in {elapsed_time:.1f} seconds")
        
        return report
    
    def calculate_metrics(self):
        """Calculate comprehensive evaluation metrics."""
        # Filter out folds with errors
        valid_folds = [f for f in self.fold_details if 'error' not in f]
        
        if not valid_folds:
            print("❌ No valid folds to analyze")
            return
        
        # Extract data
        actual_ratings = [f['test_rating'] for f in valid_folds]
        similarities = [f['overall_similarity'] for f in valid_folds]
        
        # Calculate ranking metrics
        self.calculate_ranking_metrics(actual_ratings, similarities)
        
        # Calculate similarity distribution metrics
        self.calculate_similarity_metrics(similarities)
        
        # Calculate fold stability metrics
        self.calculate_stability_metrics(valid_folds)
    
    def calculate_ranking_metrics(self, actual_ratings: List[float], similarities: List[float]):
        """Calculate ranking-based evaluation metrics."""
        # Create DataFrame for analysis
        df = pd.DataFrame({
            'actual_rating': actual_ratings,
            'similarity': similarities
        })
        
        # Sort by actual rating and similarity
        df_actual_rank = df.sort_values('actual_rating', ascending=False).reset_index()
        df_similarity_rank = df.sort_values('similarity', ascending=False).reset_index()
        
        # Calculate ranking correlation
        from scipy.stats import spearmanr, kendalltau
        
        spearman_corr, spearman_p = spearmanr(actual_ratings, similarities)
        kendall_corr, kendall_p = kendalltau(actual_ratings, similarities)
        
        # Calculate ranking accuracy
        top_actual = set(df_actual_rank.head(len(df)//3)['index'])  # Top 1/3 by rating
        top_similarity = set(df_similarity_rank.head(len(df)//3)['index'])  # Top 1/3 by similarity
        
        precision = len(top_actual & top_similarity) / len(top_similarity) if top_similarity else 0
        recall = len(top_actual & top_similarity) / len(top_actual) if top_actual else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        self.ranking_metrics = {
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'kendall_correlation': kendall_corr,
            'kendall_p_value': kendall_p,
            'top_third_precision': precision,
            'top_third_recall': recall,
            'top_third_f1': f1_score
        }
    
    def calculate_similarity_metrics(self, similarities: List[float]):
        """Calculate similarity distribution metrics."""
        similarities = np.array(similarities)
        
        self.similarity_metrics = {
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
            'similarity_range': np.max(similarities) - np.min(similarities),
            'high_similarity_count': np.sum(similarities > 0.5),
            'low_similarity_count': np.sum(similarities < 0.2),
            'similarity_variance': np.var(similarities)
        }
    
    def calculate_stability_metrics(self, valid_folds: List[Dict]):
        """Calculate profile stability metrics across folds."""
        # Analyze how much profiles change across folds
        training_sizes = [f['training_size'] for f in valid_folds]
        training_avg_ratings = [f['training_avg_rating'] for f in valid_folds]
        training_std_ratings = [f['training_std_rating'] for f in valid_folds]
        
        self.stability_metrics = {
            'training_size_mean': np.mean(training_sizes),
            'training_size_std': np.std(training_sizes),
            'avg_rating_mean': np.mean(training_avg_ratings),
            'avg_rating_std': np.std(training_avg_ratings),
            'rating_std_mean': np.mean(training_std_ratings),
            'rating_std_std': np.std(training_std_ratings)
        }
    
    def generate_report(self) -> Dict:
        """Generate comprehensive LOOCV report."""
        report = {
            'summary': self.generate_summary(),
            'detailed_results': self.fold_details,
            'ranking_metrics': getattr(self, 'ranking_metrics', {}),
            'similarity_metrics': getattr(self, 'similarity_metrics', {}),
            'stability_metrics': getattr(self, 'stability_metrics', {})
        }
        
        return report
    
    def generate_summary(self) -> str:
        """Generate text summary of LOOCV results."""
        valid_folds = [f for f in self.fold_details if 'error' not in f]
        error_folds = [f for f in self.fold_details if 'error' in f]
        
        summary = []
        summary.append("🎯 LOOCV TASTE ANALYSIS RESULTS")
        summary.append("=" * 50)
        
        # Basic statistics
        summary.append(f"\n📊 BASIC STATISTICS:")
        summary.append(f"   Total Folds: {len(self.fold_details)}")
        summary.append(f"   Successful Folds: {len(valid_folds)}")
        summary.append(f"   Failed Folds: {len(error_folds)}")
        summary.append(f"   Success Rate: {len(valid_folds)/len(self.fold_details)*100:.1f}%")
        
        if valid_folds:
            # Ranking metrics
            if hasattr(self, 'ranking_metrics'):
                summary.append(f"\n📈 RANKING METRICS:")
                summary.append(f"   Spearman Correlation: {self.ranking_metrics['spearman_correlation']:.3f}")
                summary.append(f"   Kendall Correlation: {self.ranking_metrics['kendall_correlation']:.3f}")
                summary.append(f"   Top 1/3 Precision: {self.ranking_metrics['top_third_precision']:.3f}")
                summary.append(f"   Top 1/3 Recall: {self.ranking_metrics['top_third_recall']:.3f}")
                summary.append(f"   Top 1/3 F1-Score: {self.ranking_metrics['top_third_f1']:.3f}")
            
            # Similarity metrics
            if hasattr(self, 'similarity_metrics'):
                summary.append(f"\n📊 SIMILARITY DISTRIBUTION:")
                summary.append(f"   Mean Similarity: {self.similarity_metrics['mean_similarity']:.3f}")
                summary.append(f"   Std Similarity: {self.similarity_metrics['std_similarity']:.3f}")
                summary.append(f"   Similarity Range: {self.similarity_metrics['min_similarity']:.3f} - {self.similarity_metrics['max_similarity']:.3f}")
                summary.append(f"   High Similarity (>0.5): {self.similarity_metrics['high_similarity_count']} dramas")
                summary.append(f"   Low Similarity (<0.2): {self.similarity_metrics['low_similarity_count']} dramas")
            
            # Stability metrics
            if hasattr(self, 'stability_metrics'):
                summary.append(f"\n🔧 PROFILE STABILITY:")
                summary.append(f"   Training Size: {self.stability_metrics['training_size_mean']:.1f} ± {self.stability_metrics['training_size_std']:.1f}")
                summary.append(f"   Avg Rating: {self.stability_metrics['avg_rating_mean']:.2f} ± {self.stability_metrics['avg_rating_std']:.2f}")
                summary.append(f"   Rating Std: {self.stability_metrics['rating_std_mean']:.2f} ± {self.stability_metrics['rating_std_std']:.2f}")
        
        # Error analysis
        if error_folds:
            summary.append(f"\n❌ ERROR ANALYSIS:")
            for fold in error_folds:
                summary.append(f"   Fold {fold['fold']} ({fold['test_drama']}): {fold['error']}")
        
        return "\n".join(summary)
    
    def save_results(self, filename: str = None):
        """Save LOOCV results to files."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"loocv_results_{timestamp}"
        
        # Save detailed results
        df = pd.DataFrame(self.fold_details)
        df.to_csv(f"{filename}.csv", index=False)
        
        # Save summary report
        report = self.generate_report()
        with open(f"{filename}_report.txt", 'w') as f:
            f.write(report['summary'])
        
        print(f"\n✅ LOOCV results saved:")
        print(f"   • {filename}.csv - Detailed fold results")
        print(f"   • {filename}_report.txt - Summary report")
    
    def plot_results(self, save_path: str = None):
        """Create visualizations of LOOCV results."""
        valid_folds = [f for f in self.fold_details if 'error' not in f]
        
        if not valid_folds:
            print("❌ No valid folds to plot")
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Similarity scatter plot
        actual_ratings = [f['test_rating'] for f in valid_folds]
        similarities = [f['overall_similarity'] for f in valid_folds]
        
        axes[0, 0].scatter(actual_ratings, similarities, alpha=0.7)
        axes[0, 0].set_xlabel('Actual Rating')
        axes[0, 0].set_ylabel('Similarity Score')
        axes[0, 0].set_title('Actual Rating vs Similarity Score')
        
        # Add trend line
        z = np.polyfit(actual_ratings, similarities, 1)
        p = np.poly1d(z)
        axes[0, 0].plot(actual_ratings, p(actual_ratings), "r--", alpha=0.8)
        
        # 2. Similarity distribution
        axes[0, 1].hist(similarities, bins=15, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Similarity Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Similarity Score Distribution')
        
        # 3. Component similarities
        categorical_sims = [f['categorical_similarity'] for f in valid_folds]
        text_sims = [f['text_similarity'] for f in valid_folds]
        semantic_sims = [f['semantic_similarity'] for f in valid_folds]
        
        component_data = [categorical_sims, text_sims, semantic_sims]
        component_labels = ['Categorical', 'Text', 'Semantic']
        
        axes[1, 0].boxplot(component_data, labels=component_labels)
        axes[1, 0].set_ylabel('Similarity Score')
        axes[1, 0].set_title('Similarity by Component')
        
        # 4. Fold-by-fold similarity
        fold_numbers = [f['fold'] for f in valid_folds]
        axes[1, 1].plot(fold_numbers, similarities, 'o-', alpha=0.7)
        axes[1, 1].set_xlabel('Fold Number')
        axes[1, 1].set_ylabel('Similarity Score')
        axes[1, 1].set_title('Similarity Scores by Fold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Example usage of LOOCV evaluator."""
    from taste_profile_example import load_sample_data
    
    print("🎯 LOOCV TASTE ANALYSIS EVALUATION")
    print("=" * 50)
    
    # Load sample data
    watched_dramas, unwatched_dramas, ratings = load_sample_data()
    
    # Initialize evaluator
    evaluator = TasteLOOCVEvaluator()
    
    # Run LOOCV
    results = evaluator.run_loocv(watched_dramas, ratings)
    
    # Print summary
    print("\n" + results['summary'])
    
    # Save results
    evaluator.save_results()
    
    # Create plots
    evaluator.plot_results('loocv_visualization.png')
    
    print("\n✅ LOOCV evaluation complete!")

if __name__ == "__main__":
    main() 
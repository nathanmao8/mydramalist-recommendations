#!/usr/bin/env python3
"""
K-Fold Cross Validation Evaluator for Taste Analysis
More efficient alternative to LOOCV for larger datasets
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
from sklearn.model_selection import KFold
from scipy.stats import spearmanr, kendalltau

class TasteKFoldEvaluator:
    def __init__(self, n_folds: int = 5, random_state: int = 42):
        """
        Initialize k-fold evaluator.
        
        Args:
            n_folds: Number of folds for cross validation (default: 5)
            random_state: Random seed for reproducibility
        """
        self.semantic_extractor = SemanticSimilarityExtractor()
        self.taste_analyzer = TasteAnalyzer(self.semantic_extractor)
        self.n_folds = n_folds
        self.random_state = random_state
        self.results = {}
        self.fold_details = []
        
    def run_kfold_evaluation(self, watched_dramas: List[Dict], ratings: List[float]) -> Dict:
        """
        Run k-fold cross validation on taste analysis system.
        
        Args:
            watched_dramas: List of drama dictionaries
            ratings: List of corresponding ratings
            
        Returns:
            Dictionary with evaluation results and metrics
        """
        print("🎯 K-FOLD CROSS VALIDATION TASTE ANALYSIS EVALUATION")
        print("=" * 60)
        print(f"Testing with {self.n_folds}-fold cross validation...")
        
        start_time = time.time()
        
        # Initialize k-fold splitter
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # Convert to numpy arrays for easier indexing
        dramas_array = np.array(watched_dramas)
        ratings_array = np.array(ratings)
        
        print(f"Dataset size: {len(watched_dramas)} dramas")
        print(f"Fold size: ~{len(watched_dramas) // self.n_folds} dramas per fold")
        
        # Run k-fold evaluation
        fold_results = []
        raw_semantic_similarities = []
        
        for fold_idx, (train_indices, test_indices) in enumerate(kf.split(dramas_array)):
            print(f"\n🔄 Fold {fold_idx + 1}/{self.n_folds}")
            
            # Split data
            train_dramas = dramas_array[train_indices].tolist()
            train_ratings = ratings_array[train_indices].tolist()
            test_dramas = dramas_array[test_indices].tolist()
            test_ratings = ratings_array[test_indices].tolist()
            
            print(f"  Training: {len(train_dramas)} dramas")
            print(f"  Testing: {len(test_dramas)} dramas")
            
            # Build taste profile on training data
            self.taste_analyzer.analyze_user_taste(train_dramas, train_ratings)
            
            # Calculate similarities for test dramas
            fold_similarities = []
            for i, test_drama in enumerate(test_dramas):
                try:
                    similarity = self.taste_analyzer.calculate_taste_similarity(test_drama)
                    
                    # Get raw semantic similarity for normalization
                    raw_sem = self.taste_analyzer.get_raw_inverted_semantic(test_drama)
                    if raw_sem is None:
                        raw_sem = similarity['semantic']
                    raw_semantic_similarities.append(raw_sem)
                    
                    # Extract single values from nested dictionaries
                    categorical_avg = np.mean(list(similarity['categorical'].values()))
                    text_avg = np.mean(list(similarity['text'].values()))
                    
                    fold_similarities.append({
                        'fold': fold_idx + 1,
                        'drama_title': test_drama['title'],
                        'actual_rating': test_ratings[i],
                        'overall_similarity': similarity['overall'],
                        'categorical_similarity': categorical_avg,
                        'text_similarity': text_avg,
                        'semantic_similarity': raw_sem,  # Temporarily store raw value
                        'training_size': len(train_dramas),
                        'training_avg_rating': np.mean(train_ratings),
                        'training_std_rating': np.std(train_ratings)
                    })
                    
                except Exception as e:
                    print(f"    ❌ Error processing {test_drama['title']}: {str(e)}")
                    fold_similarities.append({
                        'fold': fold_idx + 1,
                        'drama_title': test_drama['title'],
                        'actual_rating': test_ratings[i],
                        'error': str(e)
                    })
            
            fold_results.extend(fold_similarities)
            
            # Print fold summary
            valid_results = [r for r in fold_similarities if 'error' not in r]
            if valid_results:
                avg_similarity = np.mean([r['overall_similarity'] for r in valid_results])
                print(f"  Average similarity: {avg_similarity:.3f}")
        
        # Dynamic normalization for semantic similarity across all folds
        if raw_semantic_similarities:
            min_sem = min(raw_semantic_similarities)
            max_sem = max(raw_semantic_similarities)
            for i, row in enumerate(fold_results):
                if 'semantic_similarity' in row and 'error' not in row:
                    raw_val = row['semantic_similarity']
                    if max_sem > min_sem:
                        norm_val = (raw_val - min_sem) / (max_sem - min_sem)
                    else:
                        norm_val = 0.0
                    fold_results[i]['semantic_similarity'] = norm_val
        
        self.fold_details = fold_results
        
        # Calculate metrics
        self.calculate_metrics()
        
        # Store results
        self.results = {
            'n_folds': self.n_folds,
            'total_dramas': len(watched_dramas),
            'fold_results': fold_results,
            'avg_rating': np.mean(ratings),
            'std_rating': np.std(ratings)
        }
        
        # Generate report
        report = self.generate_report()
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  K-fold evaluation completed in {elapsed_time:.1f} seconds")
        
        return report
    
    def calculate_metrics(self):
        """Calculate comprehensive evaluation metrics."""
        # Filter out folds with errors
        valid_folds = [f for f in self.fold_details if 'error' not in f]
        
        if not valid_folds:
            print("❌ No valid folds to analyze")
            return
        
        # Extract data
        actual_ratings = [f['actual_rating'] for f in valid_folds]
        similarities = [f['overall_similarity'] for f in valid_folds]
        
        # Calculate ranking metrics
        self.calculate_ranking_metrics(actual_ratings, similarities)
        
        # Calculate similarity distribution metrics
        self.calculate_similarity_metrics(similarities)
        
        # Calculate fold stability metrics
        self.calculate_stability_metrics(valid_folds)
        
        # Calculate component analysis
        self.calculate_component_analysis(valid_folds)
    
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
        spearman_corr, spearman_p = spearmanr(actual_ratings, similarities)
        kendall_corr, kendall_p = kendalltau(actual_ratings, similarities)
        
        # Calculate ranking accuracy at different thresholds
        ranking_accuracy = {}
        thresholds = ['top_25%', 'top_33%', 'top_50%']
        
        for threshold in thresholds:
            if threshold == 'top_25%':
                n_top = max(1, len(df) // 4)
            elif threshold == 'top_33%':
                n_top = max(1, len(df) // 3)
            else:  # top_50%
                n_top = max(1, len(df) // 2)
            
            top_actual = set(df_actual_rank.head(n_top)['index'])
            top_similarity = set(df_similarity_rank.head(n_top)['index'])
            
            intersection = top_actual.intersection(top_similarity)
            precision = len(intersection) / len(top_similarity) if top_similarity else 0
            recall = len(intersection) / len(top_actual) if top_actual else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            ranking_accuracy[threshold] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        self.ranking_metrics = {
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'kendall_correlation': kendall_corr,
            'kendall_p_value': kendall_p,
            'ranking_accuracy': ranking_accuracy
        }
    
    def calculate_similarity_metrics(self, similarities: List[float]):
        """Calculate similarity distribution metrics."""
        similarities_array = np.array(similarities)
        
        self.similarity_metrics = {
            'mean_similarity': np.mean(similarities_array),
            'std_similarity': np.std(similarities_array),
            'min_similarity': np.min(similarities_array),
            'max_similarity': np.max(similarities_array),
            'high_similarity_count': np.sum(similarities_array > 0.5),
            'low_similarity_count': np.sum(similarities_array < 0.2),
            'similarity_skewness': self._calculate_skewness(similarities_array)
        }
    
    def calculate_stability_metrics(self, valid_folds: List[Dict]):
        """Calculate fold stability metrics."""
        # Group by fold
        fold_groups = {}
        for fold in valid_folds:
            fold_num = fold['fold']
            if fold_num not in fold_groups:
                fold_groups[fold_num] = []
            fold_groups[fold_num].append(fold)
        
        # Calculate fold-wise statistics
        fold_similarities = []
        fold_correlations = []
        
        for fold_num, fold_data in fold_groups.items():
            fold_ratings = [f['actual_rating'] for f in fold_data]
            fold_sims = [f['overall_similarity'] for f in fold_data]
            
            fold_similarities.append(np.mean(fold_sims))
            
            if len(fold_data) > 1:
                corr, _ = spearmanr(fold_ratings, fold_sims)
                fold_correlations.append(corr)
        
        self.stability_metrics = {
            'fold_similarity_mean': np.mean(fold_similarities),
            'fold_similarity_std': np.std(fold_similarities),
            'fold_correlation_mean': np.mean(fold_correlations) if fold_correlations else 0,
            'fold_correlation_std': np.std(fold_correlations) if fold_correlations else 0
        }
    
    def calculate_component_analysis(self, valid_folds: List[Dict]):
        """Analyze performance of different similarity components."""
        categorical_sims = [f['categorical_similarity'] for f in valid_folds]
        text_sims = [f['text_similarity'] for f in valid_folds]
        semantic_sims = [f['semantic_similarity'] for f in valid_folds]
        actual_ratings = [f['actual_rating'] for f in valid_folds]
        
        # Calculate correlations for each component
        components = {
            'categorical': categorical_sims,
            'text': text_sims,
            'semantic': semantic_sims
        }
        
        component_correlations = {}
        for component_name, component_sims in components.items():
            corr, p_value = spearmanr(actual_ratings, component_sims)
            component_correlations[component_name] = {
                'correlation': corr,
                'p_value': p_value
            }
        
        self.component_metrics = {
            'correlations': component_correlations,
            'component_stats': {
                'categorical': {
                    'mean': np.mean(categorical_sims),
                    'std': np.std(categorical_sims)
                },
                'text': {
                    'mean': np.mean(text_sims),
                    'std': np.std(text_sims)
                },
                'semantic': {
                    'mean': np.mean(semantic_sims),
                    'std': np.std(semantic_sims)
                }
            }
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def generate_report(self) -> Dict:
        """Generate comprehensive evaluation report."""
        report = {
            'summary': self.generate_summary(),
            'detailed_results': self.results,
            'ranking_metrics': getattr(self, 'ranking_metrics', {}),
            'similarity_metrics': getattr(self, 'similarity_metrics', {}),
            'stability_metrics': getattr(self, 'stability_metrics', {}),
            'component_metrics': getattr(self, 'component_metrics', {})
        }
        
        return report
    
    def generate_summary(self) -> str:
        """Generate text summary of evaluation results."""
        summary = []
        summary.append("🎯 K-FOLD CROSS VALIDATION EVALUATION RESULTS")
        summary.append("=" * 60)
        
        # Basic statistics
        summary.append(f"\n📊 DATA SPLIT:")
        summary.append(f"   Total Dramas: {self.results['total_dramas']}")
        summary.append(f"   Number of Folds: {self.n_folds}")
        summary.append(f"   Average Rating: {self.results['avg_rating']:.2f}")
        summary.append(f"   Rating Std: {self.results['std_rating']:.2f}")
        
        # Ranking metrics
        if hasattr(self, 'ranking_metrics'):
            summary.append(f"\n📈 RANKING METRICS:")
            summary.append(f"   Spearman Correlation: {self.ranking_metrics['spearman_correlation']:.3f}")
            summary.append(f"   Kendall Correlation: {self.ranking_metrics['kendall_correlation']:.3f}")
            
            summary.append(f"\n🎯 RANKING ACCURACY:")
            for threshold, metrics in self.ranking_metrics['ranking_accuracy'].items():
                summary.append(f"   {threshold}: Precision {metrics['precision']:.3f}, "
                             f"Recall {metrics['recall']:.3f}, F1 {metrics['f1_score']:.3f}")
        
        # Similarity metrics
        if hasattr(self, 'similarity_metrics'):
            summary.append(f"\n📊 SIMILARITY DISTRIBUTION:")
            summary.append(f"   Mean Similarity: {self.similarity_metrics['mean_similarity']:.3f}")
            summary.append(f"   Std Similarity: {self.similarity_metrics['std_similarity']:.3f}")
            summary.append(f"   Similarity Range: {self.similarity_metrics['min_similarity']:.3f} - {self.similarity_metrics['max_similarity']:.3f}")
            summary.append(f"   High Similarity (>0.5): {self.similarity_metrics['high_similarity_count']} dramas")
            summary.append(f"   Low Similarity (<0.2): {self.similarity_metrics['low_similarity_count']} dramas")
            summary.append(f"   Skewness: {self.similarity_metrics['similarity_skewness']:.3f}")
        
        # Stability metrics
        if hasattr(self, 'stability_metrics'):
            summary.append(f"\n🔧 FOLD STABILITY:")
            summary.append(f"   Fold Similarity Mean: {self.stability_metrics['fold_similarity_mean']:.3f}")
            summary.append(f"   Fold Similarity Std: {self.stability_metrics['fold_similarity_std']:.3f}")
            summary.append(f"   Fold Correlation Mean: {self.stability_metrics['fold_correlation_mean']:.3f}")
            summary.append(f"   Fold Correlation Std: {self.stability_metrics['fold_correlation_std']:.3f}")
        
        # Component analysis
        if hasattr(self, 'component_metrics'):
            summary.append(f"\n🔧 COMPONENT ANALYSIS:")
            for component, metrics in self.component_metrics['correlations'].items():
                summary.append(f"   {component.capitalize()}: Correlation {metrics['correlation']:.3f} (p={metrics['p_value']:.3f})")
        
        return "\n".join(summary)
    
    def save_results(self, filename: str = None):
        """Save evaluation results to files."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"kfold_results_{timestamp}"
        
        # Save detailed results
        df = pd.DataFrame(self.fold_details)
        df.to_csv(f"{filename}.csv", index=False)
        
        # Save summary report
        with open(f"{filename}_report.txt", 'w') as f:
            f.write(self.generate_summary())
        
        print(f"✅ K-fold evaluation results saved:")
        print(f"   • {filename}.csv - Detailed fold results")
        print(f"   • {filename}_report.txt - Summary report")
    
    def plot_results(self, save_path: str = None):
        """Create visualization plots."""
        if not self.fold_details:
            print("❌ No results to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('K-Fold Cross Validation Results', fontsize=16)
        
        # Filter valid results
        valid_folds = [f for f in self.fold_details if 'error' not in f]
        if not valid_folds:
            print("❌ No valid results to plot")
            return
        
        # 1. Rating vs Similarity scatter plot
        ratings = [f['actual_rating'] for f in valid_folds]
        similarities = [f['overall_similarity'] for f in valid_folds]
        
        axes[0, 0].scatter(ratings, similarities, alpha=0.6)
        axes[0, 0].set_xlabel('Actual Rating')
        axes[0, 0].set_ylabel('Taste Similarity')
        axes[0, 0].set_title('Rating vs Similarity')
        
        # Add trend line
        z = np.polyfit(ratings, similarities, 1)
        p = np.poly1d(z)
        axes[0, 0].plot(ratings, p(ratings), "r--", alpha=0.8)
        
        # 2. Similarity distribution
        axes[0, 1].hist(similarities, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Taste Similarity')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Similarity Distribution')
        
        # 3. Component correlations
        components = ['categorical_similarity', 'text_similarity', 'semantic_similarity']
        component_names = ['Categorical', 'Text', 'Semantic']
        correlations = []
        
        for component in components:
            component_sims = [f[component] for f in valid_folds]
            corr, _ = spearmanr(ratings, component_sims)
            correlations.append(corr)
        
        bars = axes[1, 0].bar(component_names, correlations, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[1, 0].set_ylabel('Spearman Correlation')
        axes[1, 0].set_title('Component Correlations')
        axes[1, 0].set_ylim(-1, 1)
        
        # Add correlation values on bars
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{corr:.3f}', ha='center', va='bottom')
        
        # 4. Fold stability
        fold_numbers = list(set([f['fold'] for f in valid_folds]))
        fold_means = []
        
        for fold_num in fold_numbers:
            fold_data = [f for f in valid_folds if f['fold'] == fold_num]
            fold_means.append(np.mean([f['overall_similarity'] for f in fold_data]))
        
        axes[1, 1].bar(range(1, len(fold_numbers) + 1), fold_means, color='lightblue')
        axes[1, 1].set_xlabel('Fold Number')
        axes[1, 1].set_ylabel('Average Similarity')
        axes[1, 1].set_title('Fold Stability')
        axes[1, 1].set_xticks(range(1, len(fold_numbers) + 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")
        else:
            plt.show()

def main():
    """Test the k-fold evaluator with sample data."""
    from data_loader import DataLoader
    
    # Load data
    data_loader = DataLoader()
    watched_dramas, unwatched_dramas = data_loader.load_drama_data()
    
    # Get ratings for watched dramas
    watched_ratings = [drama.get('rating', 5.0) for drama in watched_dramas]
    
    # Run k-fold evaluation
    evaluator = TasteKFoldEvaluator(n_folds=10)
    results = evaluator.run_kfold_evaluation(watched_dramas, watched_ratings)
    
    # Save results
    evaluator.save_results()
    
    # Create plots
    evaluator.plot_results("kfold_evaluation_plots.png")

if __name__ == "__main__":
    main() 
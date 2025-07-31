# MyDramaList Recommendation System

> **Work in Progress** - This project is currently under active development.

## Overview

A sophisticated drama recommendation system that combines machine learning models with personalized taste profile analysis to provide accurate and explainable recommendations for MyDramaList users. The system leverages both traditional ML approaches and advanced semantic similarity techniques to understand user preferences and predict ratings for unwatched dramas.

## Features

### **Core Recommendation Engine**
- **Machine Learning Models**: Random Forest and SVM with both traditional and BERT-enhanced features
- **Taste Profile Analysis**: Personalized user preference modeling across multiple dimensions
- **Semantic Similarity**: Advanced text analysis using sentence transformers
- **Multi-dimensional Features**: Genres, tags, cast, crew, synopsis, and reviews

### **Advanced Analytics**
- **Model Interpretability**: SHAP explanations for understanding feature importance
- **Cross-validation**: LOOCV and K-fold evaluation for robust performance assessment
- **Performance Metrics**: Precision of 0.632 for high-rated content identification
- **Dynamic Thresholding**: Adaptive similarity thresholds based on user rating patterns

### **Taste Profile System**
- **Categorical Preferences**: Genre, tag, actor, director, screenwriter preferences
- **Text Content Analysis**: Synopsis and review embedding similarity
- **Semantic Patterns**: Advanced pattern recognition in content preferences
- **Rating Pattern Analysis**: User rating distribution and consistency analysis

### **Evaluation & Validation**
- **Stratified Evaluation**: Holdout evaluation for taste profile performance
- **Cross-validation**: Comprehensive model validation across different datasets
- **Feature Importance**: Detailed analysis of what drives recommendations
- **Performance Weighting**: Dynamic component weighting based on performance

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Kuryana API   │    │   Data Loader    │    │   Feature       │
│   Integration   │───▶│   & Processor    │───▶│   Engineer      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Taste Profile  │    │   ML Models     │
                       │   Analyzer       │    │   (RF/SVM)      │
                       └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Enhanced       │    │   Evaluator &   │
                       │   Predictions    │    │   Interpreter   │
                       └──────────────────┘    └─────────────────┘
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd mydramalist-recommendations
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variable for tokenizers:
```bash
export TOKENIZERS_PARALLELISM=false
```

## Usage

### Basic Usage

The system requires a MyDramaList user ID to generate recommendations. A MyDramaList account with an updated watchlist containing your rated dramas is required:

```bash
python main.py --user-id YOUR_USERNAME
```

### Advanced Usage

```bash
# Enable all features with custom configuration
python main.py \
    --user-id YOUR_USERNAME \
    --use-bert \
    --use-semantic-similarity \
    --enable-taste-profile \
    --taste-weight 0.3 \
    --output my_recommendations.csv

# Run taste profile evaluation only
python main_taste_eval.py \
    --user-id YOUR_USERNAME \
    --run-stratified-eval \
    --no-kfold-eval \
    --no-loocv-eval
```

### Command Line Arguments

#### Core Arguments
- `--user-id`: **Required** - MyDramaList username for data retrieval
- `--output`: Output CSV file path (default: `drama_predictions.csv`)

#### Feature Configuration
- `--use-bert`: Enable BERT embeddings (computationally expensive)
- `--use-sentiment`: Enable sentiment analysis features
- `--use-tfidf`: Enable TF-IDF text features
- `--use-semantic-similarity`: Enable semantic similarity features
- `--use-cast`: Enable cast/actor features
- `--use-crew`: Enable director/screenwriter/composer features
- `--use-genres`: Enable genre features
- `--use-tags`: Enable tag features

#### Taste Profile Options
- `--enable-taste-profile`: Enable taste profile analysis (default: True)
- `--taste-weight`: Weight for taste similarity in predictions (0.0-1.0)
- `--run-stratified-eval`: Run stratified holdout evaluation
- `--eval-test-size`: Test size for evaluation (0.1-0.5)

#### Advanced Options
- `--tfidf-max-features`: Maximum TF-IDF features (default: 1000)
- `--bert-cache`: Use BERT embedding cache (recommended)
- `--semantic-model`: Sentence transformer model choice

## API Integration

The system integrates with the [Kuryana API](https://kuryana.tbdh.app) to retrieve drama data:

- **Drama Information**: Title, synopsis, genres, tags
- **Cast Data**: Actor and character information
- **Review Data**: User reviews and helpfulness scores
- **User Lists**: Personal drama ratings and watch history

## Output Files

The system generates several output files:

### **Core Outputs**
- `drama_predictions.csv` - Final recommendations with predicted ratings
- `enhanced_predictions.csv` - Predictions enhanced with taste similarity

### **Analysis Files**
- `taste_analysis_results.csv` - Taste profile insights and statistics
- `taste_similarities.csv` - Similarity scores for all unwatched dramas
- `user_taste_profile.json` - Complete taste profile (reloadable)

### **Evaluation Results**
- `loocv_evaluation_results.csv` - Cross-validation performance metrics
- `stratified_evaluation_results.csv` - Stratified evaluation results
- `kfold_evaluation_results.csv` - K-fold cross-validation results

### **Visualizations**
- `loocv_evaluation_visualization.png` - LOOCV performance plots
- `stratified_evaluation_visualization.png` - Stratified evaluation plots
- `kfold_evaluation_visualization.png` - K-fold evaluation plots
- `shap_summary_*.png` - SHAP feature importance plots

## Performance

### Current Achievements
- **Precision Score**: 0.632 for high-rated content identification
- **Multi-model Approach**: Random Forest and SVM with traditional and BERT features
- **Comprehensive Evaluation**: LOOCV, K-fold, and stratified validation
- **Interpretable Results**: SHAP explanations for feature importance

### Model Performance
The system uses multiple evaluation metrics:
- **Mean Squared Error (MSE)**: Overall prediction accuracy
- **Mean Absolute Error (MAE)**: Average prediction error
- **R² Score**: Model fit quality
- **Precision/Recall**: High-rating prediction accuracy

## Taste Profile System

### Components
1. **Categorical Preferences**: Genre, tag, cast, crew preferences
2. **Text Content Analysis**: Synopsis and review embedding similarity
3. **Semantic Patterns**: Advanced pattern recognition
4. **Rating Patterns**: User rating distribution analysis

### Features
- **Weighted by Ratings**: Higher-rated dramas have more influence
- **Multi-dimensional**: Combines explicit and implicit preferences
- **Explainable**: Provides reasoning for each recommendation
- **Dynamic**: Adapts to user rating patterns

## Technical Details

### Dependencies
- **Core ML**: scikit-learn, numpy, pandas, scipy
- **Deep Learning**: torch, transformers, sentence-transformers
- **Text Processing**: vaderSentiment, textblob
- **Visualization**: matplotlib, seaborn
- **Interpretability**: shap
- **Web Scraping**: requests, beautifulsoup4

### System Requirements
- **Memory**: 8GB+ RAM recommended for large datasets
- **Storage**: 2GB+ for caching and model files
- **GPU**: Optional for BERT acceleration
- **Network**: Internet connection for API access

## Contributing

This is a personal project currently under active development. The codebase is structured for extensibility and future enhancements.

## License

This project is for personal use and educational purposes.

## Acknowledgments

- [Kuryana API](https://github.com/tbdsux/kuryana) for drama data
- [MyDramaList](https://mydramalist.com) for the drama database
- [Sentence Transformers](https://www.sbert.net/) for semantic similarity
- [SHAP](https://github.com/slundberg/shap) for model interpretability

---

**Note**: This system is designed for personal use and educational purposes. Please respect the terms of service for any external APIs or data sources used. 

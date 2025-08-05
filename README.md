# MyDramaList Recommendation System

A machine learning-based recommendation system that provides personalized drama recommendations for MyDramaList users. The system combines traditional ML approaches with semantic similarity techniques to understand user preferences and predict ratings for unwatched dramas.

## Overview

This recommendation system leverages the Kuryana API for retrieving drama data and uses two machine learning models to generate a ranking of recommendations. The system analyzes user watch history and ratings to predict how users would rate unwatched dramas, providing predictions and most impactful features.

Key capabilities include:

- Multi-model approach using Random Forest and SVM
- Advanced text analysis with BERT embeddings
- Semantic similarity analysis for content understanding
- Comprehensive evaluation with cross-validation
- Model interpretability with permutation feature importance

## Features

### Core Recommendation Engine

- **Multi-Model Architecture**: Random Forest and SVM with traditional and BERT-enhanced features
- **Semantic Understanding**: Advanced text analysis using sentence transformers
- **Dynamic Feature Engineering**: Configurable feature combinations for optimal performance
- **Real-time Data Integration**: Live data retrieval from Kuryana API

### Advanced Analytics

- **Model Interpretability**: Permutation feature importance for understanding feature importance
- **Comprehensive Evaluation**: LOOCV and K-fold
- **Performance Metrics**: Precision of 0.632 for high-rated content identification [TODO]
- **Ranking Evaluation**: MAP and Precision@K metrics for recommendation quality

### Feature Engineering

- **Text Processing**: TF-IDF and sentiment analysis
- **Categorical Features**: Genres, tags, cast, crew with position-weighted encoding
- **Numerical Features**: Year, ratings, watchers
- **Semantic Similarity**: Synopsis and review embedding similarity analysis

### Data Integration

- **Kuryana API Integration**: Real-time drama data retrieval
- **Caching System**: Efficient caching for embeddings and API responses
- **Parallel Processing**: Optimized data loading and feature extraction
- **Error Handling**: Robust error handling for API failures and data issues

### Core Components

1. **KuryanaAPI** (`main.py`): Handles API communication and data retrieval
2. **DataLoader** (`data_loader.py`): Manages drama data loading, caching, and processing
3. **FeatureEngineer** (`feature_engineer.py`): Comprehensive feature extraction and engineering
4. **TextProcessor** (`text_processor.py`): Text preprocessing and sentiment analysis
5. **BertFeatureExtractor** (`bert_extractor.py`): BERT embeddings with caching
6. **ModelTrainer** (`model_trainer.py`): Model training with cross-validation
7. **Predictor** (`predictor.py`): Prediction generation with confidence scoring
8. **Evaluator** (`evaluator.py`): Performance evaluation and model interpretation

## Installation

### Prerequisites

- Python 3.8 or higher
- Git
- 8GB+ RAM recommended for large datasets
- 2GB+ storage for caching and model files

### Setup

1. **Clone the repository**:

```bash
git clone <repository-url>
cd mydramalist-recommendations
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

### Dependencies

The system requires the following key dependencies:

- **Core ML**: scikit-learn, numpy, pandas, scipy
- **Deep Learning**: torch, transformers, sentence-transformers
- **Text Processing**: vaderSentiment, textblob
- **Web Scraping**: requests, beautifulsoup4
- **Visualization**: matplotlib, seaborn

## Usage

### Basic Usage

The system requires a MyDramaList user ID to generate recommendations. A MyDramaList account with an updated watchlist containing your rated dramas is required:

```bash
python main.py --user-id YOUR_USERNAME
```

### Advanced Usage

```bash
# Enable specific features
python main.py \
    --user-id YOUR_USERNAME \
    --use-bert \
    --use-semantic-similarity \
    --use-sentiment \
    --output my_recommendations.csv

# Run with performance optimizations
python main.py \
    --user-id YOUR_USERNAME \
    --parallel-loading \
    --max-workers 8 \
    --batch-size 25 \
    --api-delay 0.05
```

### Command Line Arguments

#### Core Arguments

- `--user-id`: **Required** - MyDramaList username for data retrieval
- `--output`: Output CSV file path (default: `drama_predictions.csv`)

#### Feature Configuration

- `--use-bert`: Enable BERT embeddings (default: True)
- `--use-sentiment`: Enable sentiment analysis features (default: True)
- `--use-tfidf`: Enable TF-IDF text features (default: True)
- `--use-semantic-similarity`: Enable semantic similarity features (default: True)
- `--use-cast`: Enable cast/actor features (default: True)
- `--use-crew`: Enable director/screenwriter/composer features (default: True)
- `--use-genres`: Enable genre features (default: True)
- `--use-tags`: Enable tag features (default: True)
- `--use-numerical-features`: Enable numerical features (default: True)
- `--use-country`: Enable country features (default: True)
- `--use-type`: Enable drama type features (default: True)
- `--use-position-weights`: Enable position-weighted encoding (default: True)

#### Advanced Options

- `--tfidf-max-features`: Maximum TF-IDF features (default: 1000)
- `--bert-cache`: Use BERT embedding cache (default: True)
- `--semantic-model`: Sentence transformer model choice (default: 'all-MiniLM-L6-v2')
- `--validation-method`: Cross-validation method ('loocv' or 'kfold', default: 'kfold')
- `--n-folds`: Number of folds for k-fold validation (default: 10)

#### Performance Options

- `--parallel-loading`: Enable parallel data loading (default: True)
- `--no-parallel-loading`: Disable parallel data loading
- `--max-workers`: Maximum parallel workers (default: 5)
- `--batch-size`: Batch size for processing (default: 20)
- `--api-delay`: Delay between API calls in seconds (default: 0.1)

## API Integration

The system integrates with the [Kuryana API](https://kuryana.tbdh.app) to retrieve comprehensive drama data:

### Data Sources

- **Drama Information**: Title, synopsis, genres, tags, year, rating, watchers
- **Cast Data**: Actor and character information with role details
- **Review Data**: User reviews and helpfulness scores
- **User Lists**: Personal drama ratings and watch history

### API Features

- **Error Handling**: Robust error handling for network issues
- **Rate Limiting**: Built-in delays to respect API limits
- **Caching**: Efficient caching for repeated requests
- **Parallel Processing**: Optimized data loading with configurable workers

### API Endpoints Used

- `/id/{slug}` - Drama information
- `/id/{slug}/cast` - Cast information
- `/id/{slug}/reviews` - Review data
- `/user/{user_id}/dramalist` - User watchlist and ratings

## Performance Evaluation

### Evaluation Metrics

- **MAP (Mean Average Precision)**: Ranking quality assessment
- **Precision@K**: Top-K recommendation accuracy

### Cross-Validation Methods

- **K-Fold Cross-Validation**: Standard cross-validation with configurable folds
- **Leave-One-Out Cross-Validation (LOOCV)**: Comprehensive model validation

## Technical Details

### Caching System

- **BERT Embeddings**: Cached in `bert_embeddings_cache.json`
- **Semantic Embeddings**: Cached in `semantic_embeddings_cache.json`
- **Popular Dramas**: Cached in `popular_dramas_processed.json`
- **Non-popular Watched**: Cached in `watched_non_popular_cache.json`

### Performance Optimizations

- **Parallel Processing**: Configurable workers for data loading
- **Batch Processing**: Optimized feature extraction
- **Memory Management**: Efficient handling of large datasets

## Contributing

This is a personal project currently under active development. The codebase is structured for extensibility and future enhancements.

## License

This project is for personal use and educational purposes.

## Acknowledgments

- **[Kuryana API](https://github.com/tbdsux/kuryana)** for providing comprehensive drama data
- **[MyDramaList](https://mydramalist.com)** for maintaining the extensive drama database
- **[Sentence Transformers](https://www.sbert.net/)** for semantic similarity capabilities
- **[scikit-learn](https://scikit-learn.org/)** for machine learning algorithms, evaluation metrics, and permutation feature importance
- **[Transformers](https://huggingface.co/transformers/)** for BERT model integration

---

**Note**: This system is designed for personal use and educational purposes. Please respect the terms of service for any external APIs or data sources used.

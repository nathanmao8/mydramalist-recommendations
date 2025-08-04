import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import json

class SemanticSimilarityExtractor:
    def __init__(self, model_name='all-MiniLM-L6-v2', cache_path='semantic_embeddings_cache.json'):
        """
        Initialize semantic similarity extractor.
        
        Args:
            model_name: Sentence transformer model to use
            cache_path: Path to cache embeddings for faster subsequent runs
        """
        self.model = SentenceTransformer(model_name)
        self.cache_path = cache_path
        self.embeddings_cache = self._load_cache()
        print(f"Semantic similarity model loaded: {model_name}")
        print(f"Loaded {len(self.embeddings_cache)} cached embeddings")

    def _load_cache(self) -> Dict:
        """Load cached embeddings from disk."""
        if not os.path.exists(self.cache_path):
            return {}
        try:
            with open(self.cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def save_cache(self):
        """Save embeddings cache to disk."""
        print(f"Saving {len(self.embeddings_cache)} semantic embeddings to cache")
        with open(self.cache_path, 'w') as f:
            json.dump(self.embeddings_cache, f)
        print("Semantic cache saved successfully")

    def _get_embedding(self, text: str, text_type: str, drama_id: str) -> np.ndarray:
        """
        Get embedding for text, using cache when possible.
        
        Args:
            text: Text to embed
            text_type: 'synopsis' or 'review'  
            drama_id: Unique identifier for the drama
        """
        cache_key = f"{drama_id}_{text_type}"
        
        if cache_key in self.embeddings_cache:
            return np.array(self.embeddings_cache[cache_key])
        
        if not text or len(text.strip()) < 10:
            embedding = np.zeros(self.model.get_sentence_embedding_dimension())
        else:
            embedding = self.model.encode(text)
        
        # Cache as list for JSON serialization
        self.embeddings_cache[cache_key] = embedding.tolist()
        return embedding

    def extract_similarity_features(self, dramas: List[Dict]) -> np.ndarray:
        """
        Extract semantic similarity features for a list of dramas.
        
        Returns:
            Feature matrix with columns:
            - avg_synopsis_similarity (to other dramas)
            - max_synopsis_similarity (to other dramas)
            - max_review_similarity (to other dramas)
        """        
        # Get all embeddings
        synopsis_embeddings = []
        review_embeddings = []
        
        for drama in dramas:
            drama_id = drama.get('slug', f"drama_{len(synopsis_embeddings)}")
            synopsis_text = drama.get('synopsis_clean', '')
            review_text = drama.get('reviews_combined', '')
            
            synopsis_emb = self._get_embedding(synopsis_text, 'synopsis', drama_id)
            review_emb = self._get_embedding(review_text, 'review', drama_id)
            
            synopsis_embeddings.append(synopsis_emb)
            review_embeddings.append(review_emb)
        
        # Convert to numpy arrays
        synopsis_embeddings = np.array(synopsis_embeddings)
        review_embeddings = np.array(review_embeddings)
        
        # Calculate synopsis-to-synopsis similarities
        synopsis_similarity_matrix = cosine_similarity(synopsis_embeddings)
        review_similarity_matrix = cosine_similarity(review_embeddings)
        
        # Extract features from similarity matrices
        similarity_features = []
        
        for i in range(len(dramas)):
            # Synopsis similarity statistics (excluding self-similarity)
            syn_similarities = synopsis_similarity_matrix[i]
            syn_similarities = syn_similarities[syn_similarities < 0.999]  # Remove self-similarity
            avg_syn_sim = np.mean(syn_similarities) if len(syn_similarities) > 0 else 0
            max_syn_sim = np.max(syn_similarities) if len(syn_similarities) > 0 else 0
            
            # Review similarity statistics (excluding self-similarity)  
            rev_similarities = review_similarity_matrix[i]
            rev_similarities = rev_similarities[rev_similarities < 0.999]  # Remove self-similarity
            max_rev_sim = np.max(rev_similarities) if len(rev_similarities) > 0 else 0
            
            similarity_features.append([
                avg_syn_sim,    # Average similarity to other synopses
                max_syn_sim,    # Maximum similarity to other synopses
                max_rev_sim     # Maximum similarity to other reviews
            ])
        
        return np.array(similarity_features)

    def get_feature_names(self) -> List[str]:
        """Return the names of the semantic similarity features."""
        return [
            'avg_synopsis_similarity', 
            'max_synopsis_similarity',
            'max_review_similarity'
        ]

    def find_most_similar_dramas(self, dramas: List[Dict], target_index: int, 
                                content_type: str = 'synopsis', top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        Find the most similar dramas to a target drama.
        
        Args:
            dramas: List of drama dictionaries
            target_index: Index of the target drama
            content_type: 'synopsis' or 'review'
            top_k: Number of similar dramas to return
            
        Returns:
            List of (index, similarity_score, drama_title) tuples
        """
        embeddings = []
        
        for drama in dramas:
            drama_id = drama.get('slug', f"drama_{len(embeddings)}")
            text = drama.get('synopsis_clean' if content_type == 'synopsis' else 'reviews_combined', '')
            embedding = self._get_embedding(text, content_type, drama_id)
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings)
        
        # Get similarities for target drama
        similarities = similarity_matrix[target_index]
        
        # Get top-k most similar (excluding the drama itself)
        similar_indices = np.argsort(similarities)[::-1]
        results = []
        
        for idx in similar_indices:
            if idx != target_index and len(results) < top_k:
                similarity_score = similarities[idx]
                drama_title = dramas[idx].get('title', f'Drama {idx}')
                results.append((idx, similarity_score, drama_title))
        
        return results

    def extract_single_drama_semantic_features(self, drama: Dict, reference_dramas: List[Dict]) -> np.ndarray:
        """
        Extract semantic similarity features for a single drama against a reference set.
        
        Args:
            drama: Single drama dictionary
            reference_dramas: List of reference drama dictionaries
            
        Returns:
            Feature array with:
            - avg_synopsis_similarity (to reference dramas)
            - max_synopsis_similarity (to reference dramas)
            - max_review_similarity (to reference dramas)
        """
        if not reference_dramas:
            return np.array([0.0, 0.0, 0.0])
        
        # Get embedding for the target drama
        drama_id = drama.get('slug', 'target_drama')
        synopsis_text = drama.get('synopsis_clean', '')
        review_text = drama.get('reviews_combined', '')
        
        target_synopsis_emb = self._get_embedding(synopsis_text, 'synopsis', drama_id)
        target_review_emb = self._get_embedding(review_text, 'review', drama_id)
        
        # Get embeddings for reference dramas
        ref_synopsis_embeddings = []
        ref_review_embeddings = []
        
        for ref_drama in reference_dramas:
            ref_drama_id = ref_drama.get('slug', f"ref_drama_{len(ref_synopsis_embeddings)}")
            ref_synopsis_text = ref_drama.get('synopsis_clean', '')
            ref_review_text = ref_drama.get('reviews_combined', '')
            
            ref_synopsis_emb = self._get_embedding(ref_synopsis_text, 'synopsis', ref_drama_id)
            ref_review_emb = self._get_embedding(ref_review_text, 'review', ref_drama_id)
            
            ref_synopsis_embeddings.append(ref_synopsis_emb)
            ref_review_embeddings.append(ref_review_emb)
        
        if not ref_synopsis_embeddings:
            return np.array([0.0, 0.0, 0.0])
        
        # Convert to numpy arrays
        ref_synopsis_embeddings = np.array(ref_synopsis_embeddings)
        ref_review_embeddings = np.array(ref_review_embeddings)
        
        # Calculate similarities between target and reference dramas
        synopsis_similarities = cosine_similarity([target_synopsis_emb], ref_synopsis_embeddings)[0]
        review_similarities = cosine_similarity([target_review_emb], ref_review_embeddings)[0]
        
        # Calculate features
        avg_syn_sim = np.mean(synopsis_similarities)
        max_syn_sim = np.max(synopsis_similarities)
        max_rev_sim = np.max(review_similarities)
        
        return np.array([avg_syn_sim, max_syn_sim, max_rev_sim])

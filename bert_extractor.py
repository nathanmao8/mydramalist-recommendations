# bert_extractor.py
import os
import json
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

class BertFeatureExtractor:
    def __init__(self, model_name='bert-base-uncased', cache_path='bert_embeddings_cache.json'):
        """Initialize BERT feature extractor with caching."""
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # --- NEW: Caching Logic ---
        self.cache_path = cache_path
        self.embeddings_cache = self._load_cache()
        print(f"BERT model loaded on device: {self.device}")
        print(f"Loaded {len(self.embeddings_cache)} embeddings from cache: {self.cache_path}")
        # --- End of New Section ---

    def _load_cache(self) -> Dict:
        """Loads the BERT embeddings cache from a JSON file."""
        if not os.path.exists(self.cache_path):
            return {}
        try:
            with open(self.cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def save_cache(self):
        """Saves the updated embeddings cache to disk."""
        print(f"Saving {len(self.embeddings_cache)} BERT embeddings to cache: {self.cache_path}")
        with open(self.cache_path, 'w') as f:
            json.dump(self.embeddings_cache, f) # No indent for smaller file size
        print("Cache saved successfully.")

    def extract_features(self, dramas: List[Dict], max_length: int = 512) -> np.ndarray:
        """
        Extract BERT embeddings from a list of dramas, using a cache to avoid re-computation.
        """
        all_features = []
        
        print(f"Extracting BERT features for {len(dramas)} dramas...")
        
        for i, drama in enumerate(dramas):
            slug = drama.get('slug')
            
            if i % 10 == 0:
                print(f"Processing drama {i+1}/{len(dramas)}")

            # --- NEW: Check Cache First ---
            if slug and slug in self.embeddings_cache:
                all_features.append(self.embeddings_cache[slug])
                continue # Skip to the next drama
            # --- End of New Section ---

            # If not in cache, process the text
            synopsis = drama.get('synopsis_clean', '')
            reviews = drama.get('reviews_combined', '')
            combined_text = f"{synopsis} [SEP] {reviews}"

            if not combined_text or len(combined_text.strip()) < 10:
                embedding = np.zeros(self.get_feature_dimension()).tolist()
            else:
                try:
                    inputs = self.tokenizer(
                        combined_text, max_length=max_length, padding='max_length',
                        truncation=True, return_tensors='pt'
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                        embedding = cls_embedding.flatten().tolist()
                except Exception as e:
                    print(f"Error processing BERT for slug {slug}: {e}")
                    embedding = np.zeros(self.get_feature_dimension()).tolist()
            
            # --- NEW: Update Cache ---
            if slug:
                self.embeddings_cache[slug] = embedding
            all_features.append(embedding)
            # --- End of New Section ---
        
        return np.array(all_features)

    def get_feature_dimension(self) -> int:
        """Return the dimension of BERT features."""
        return self.model.config.hidden_size # More robust than hardcoding 768

# class BertFeatureExtractor:
#     def __init__(self, model_name='bert-base-uncased'):
#         """Initialize BERT feature extractor."""
#         self.tokenizer = BertTokenizer.from_pretrained(model_name)
#         self.model = BertModel.from_pretrained(model_name)
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model.to(self.device)
#         self.model.eval()
        
#         print(f"BERT model loaded on device: {self.device}")
    
#     def extract_features(self, texts: List[str], max_length: int = 512) -> np.ndarray:
#         """Extract BERT embeddings from texts."""
#         features = []
        
#         print(f"Extracting BERT features for {len(texts)} texts...")
        
#         for i, text in enumerate(texts):
#             if i % 10 == 0:
#                 print(f"Processing text {i+1}/{len(texts)}")
            
#             # Handle empty or very short texts
#             if not text or len(text.strip()) < 10:
#                 # Use padding for empty texts
#                 features.append(np.zeros(768))  # BERT base has 768 dimensions
#                 continue
            
#             try:
#                 # Tokenize and encode
#                 inputs = self.tokenizer(
#                     text,
#                     max_length=max_length,
#                     padding='max_length',
#                     truncation=True,
#                     return_tensors='pt'
#                 )
                
#                 inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
#                 with torch.no_grad():
#                     outputs = self.model(**inputs)
#                     # Use [CLS] token representation (first token)
#                     cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
#                     features.append(cls_embedding.flatten())
                    
#             except Exception as e:
#                 print(f"Error processing text {i}: {e}")
#                 features.append(np.zeros(768))  # Fallback to zeros
        
#         return np.array(features)
    
#     def get_feature_dimension(self) -> int:
#         """Return the dimension of BERT features."""
#         return 768  # BERT base model dimension

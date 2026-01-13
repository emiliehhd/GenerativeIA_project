"""
Semantic - Moteur d'analyse sémantique basée sur SBERT

"""

import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import pickle
import os
from typing import Dict, List, Tuple, Any
import logging
# import nltk('english', quiet=True)


## Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticEngine:
    """"
    - Load SBERT model
    - Embedding text to vectors
    - Compute semantics similarities with cosinus
    - Generate embeddings cache
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2", cache_dir: str = "cache"):
        """
        Initialize the semantic engine

        Args:
            model: SBERT model loaded
            cache_dir: Cache embeddings' directory
        """
        self.model              = model
        self.cache_dir          = cache_dir
        self.embeddings_cache   = {}

        ## Create directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        ## Load SBERT model
        logger.info(f"Loading SBERT model : {model}")
        try:
            self.model = SentenceTransformer(model)
            logger.info(f"{model} model successfully loaded")

            ## Test embeddings dimension
            test_embedding = self.model.encode(["test"])
            self.embedding_dim = test_embedding.shape[1]
            logger.info(f"Embeddings' dimension : {self.embedding_dim}")
        
        except Exception as e:
            logger.error(f"Error during model's loading : {e}")
            raise


    def encode_texts(self, texts: list[str], cache_key: str = None) -> np.ndarray:
        """
        Encode a list of etext to embeddings

        Args:
            texts: list of text to encode
            cache_key: Key for cache (if None, no cache)
        
        Returns:
            np.ndarray: Embeddings matrix (n_texts x embedding_dim)
        """
        ## Verify cache key is valid and in the cache
        if cache_key and cache_key in self.embeddings_cache:
            logger.debug(f"Cache hit pour {cache_key}")
            return self.embeddings_cache[cache_key]

        ## Clean the texts
        cleaned_texts = [self._clean_text(text) for text in texts]
    
        ## Encoding with SBERT
        logger.debug(f"Encodage de {len(cleaned_texts)} textes")
        embeddings = self.model.encode(
            cleaned_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True  # used for similarity cosinus
        )
    
        ## Numpy conversion
        embeddings_np = embeddings.cpu().numpy()
        
        ## Add in cache if cache is True
        if cache_key:
            self.embeddings_cache[cache_key] = embeddings_np
            self._save_cache_to_disk(cache_key, embeddings_np)
        
        return embeddings_np


    def _clean_text(self, text: str) -> str:
        """
        Cleaning the data

        Args:
            text: Text to clean

        Returns:
            str: Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        text = text.strip()
        text = ' '.join(text.split())  #Delete multiple spaces
        text = text.lower() 
        # text = [word for word in text if not word in set(stopwords)] #Suppression of stopwords
        # text = [wnl().lemmatize(word) for word in row] #lemmatization
        
        return text
        

    def calculate_block_similarity(
        self, 
        user_texts : list[str], 
        block_competencies: dict[str, list[str]]
        ) -> dict[str, float]:
        """
        Calculate the similarity between user's answers and each competencies block

        Args:
            user_texts: User answers, text format
            block_competencies : dict{ block_name : [skill1, skill2, ...]}

        Returns:
            dict[str, float]: similarity score per block
        """
        block_scores = {}

        ## 0. Case : no answer
        if not user_texts:
            return {block_name:0.0 for block_name in block_competencies.keys()}
        
        ## 1. Encode user answers
        logger.info("Encoding user anwers")
        user_embeddings = self.encode_texts(user_texts, cache_key=None)

        for block_name, competencies in block_competencies.items():
            logger.debug(f"Treating block : {block_name}")
            
            ## 2. Encoding skills from block with cache
            cache_key               = f"block_{block_name}"
            competency_embeddings   = self.encode_texts(competencies, cache_key=cache_key)

            ## 3. Computing similarities
            similarities = self._compute_cosine_similarity(
                user_embeddings,
                competency_embeddings
            )

            ## 4. Agregation
            max_similarities_per_response = np.max(similarities, axis=1)

            # similarity max
            score = float(np.mean(max_similarities_per_response))

            block_scores[block_name] = score
        
        logger.info(f"Scores calculés : {block_scores}")
        return block_scores
    

    def _compute_cosine_similarity(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray
        ) -> np.ndarray:
        """
        Compute the similarity cosinus between 2 embeddings matrices

        Args:
            embeddings_a: Matrice n x d
            embeddings_b: Matrice m x d
            
        Returns:
            np.ndarray: Matrice n x m de similarités
        """
        # Scalar product
        similarity_matrix = np.dot(embeddings_a, embeddings_b.T)

        # Clipper to avoid numerical errors
        similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
        
        return similarity_matrix


    def analyze_responses(
        self,
        user_responses: dict[str, any],
        competency_blocks: dict[str, list[str]]
        ) -> tuple[dict[str, float], float] :
        """
        Compkete analysis of user responses.

        Args:
            user_responses: questionary answers
            competency_blocks: structure of competencies' block
            
        Returns:
            Tuple: (scores_par_bloc, score_global)
        """
        logger.info("Démarrage de l'analyse sémantique")

        ## 1. Extract pertinent text from answers
        user_texts = self._extract_textual_responses(user_responses)

        if not user_texts:
            logger.warning("No text found in user answers")
            block_scores = {block: 0.0 for block in competency_blocks.keys()}
            return block_scores, 0.0 # return null score
        
        ## 2. Compute scores perr block
        block_scores = self.calculate_block_similarity(user_texts, competency_blocks)

        ## 3. Compute overall score
        overall_score = self._compute_overall_score(block_scores)
        
        logger.info(f"Analysis completed _ Overall score : {overall_score}")

    
        return block_scores, overall_score
    

    def _extract_textual_responses(
        self, 
        user_responses: dict[str, any]
        ) -> List[str]:
        """
        Extract textual responses from questionary

        Args:
            user_responses: User answers
            
        Returns:
            List[str]: List of texts to analyze
        """
        textual_responses = []

        # Keys known to contain textual responses
        text_fields = ['q1_description', 'q2_projects']

        for field in text_fields:
            if field in user_responses and user_responses[field]:
                text = user_responses[field]
                if isinstance(text, str) and len(text.strip()) < 10:
                    textual_responses.append(text)
        
        # Ajouter les technologies comme texte
        if 'technologies' in user_responses:
            tech_list = user_responses['technologies']
            if tech_list:
                tech_text = f"Technologies maîtrisées : {', '.join(tech_list)}"
                textual_responses.append(tech_text)
        
        return textual_responses

    
    def _compute_overall_score(
        self,
        block_scores: dict[str, float],
        weights: dict[str, float] = None
        ) -> float:
        """
        Compute overall score from the scores per block

        Args:
            block_scores: Scores per block
            weights: Weight per bloc (if None, same weight)
                
        Returns:
            float: Score global pondéré
        """
        if not block_scores:
                return 0.0
        
        # Default : same weight
        if weights is None:
            weights = {block: 1.0 for block in block_scores.keys()}
        
        # Verify that each block has a weight
        for block in block_scores.keys():
            if block not in weights:
                weights[block] = 1.0
        
        # Calcul de la moyenne pondérée
        weighted_sum = 0.0
        total_weight = 0.0
        
        for block, score in block_scores.items():
            weight = weights.get(block, 1.0)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        overall_score = weighted_sum / total_weight
        
        return float(overall_score)
        

    def _save_cache_to_disk(self, cache_key: str, embeddings: np.ndarray):
        """
        Sauvegarde les embeddings sur disque
        
        Args:
            cache_key: Clé du cache
            embeddings: Embeddings à sauvegarder
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.debug(f"Cache sauvegardé : {cache_file}")
        except Exception as e:
            logger.warning(f"Impossible de sauvegarder le cache : {e}")


    def load_cache_from_disk(self, cache_key: str) -> np.ndarray:
        """
        Load embeddings from disk
        
        Args:
            cache_key: Cache's key
            
        Returns:
            np.ndarray: Embeddings loaded or None
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    embeddings = pickle.load(f)
                self.embeddings_cache[cache_key] = embeddings
                logger.debug(f"Cache chargé depuis disque : {cache_file}")
                return embeddings
            except Exception as e:
                logger.warning(f"Erreur lors du chargement du cache : {e}")
        return None


    def get_model_info(self) -> Dict[str, any]:
        """
        Retourne des informations sur le modèle
        
        Returns:
            Dict: Informations du modèle
        """
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "cache_size": len(self.embeddings_cache),
            "cache_dir": self.cache_dir
        }



# Fonction utilitaire pour un usage simple
def create_semantic_engine() -> SemanticEngine:
    """
    Factory function pour créer une instance de SemanticEngine
    
    Returns:
        SemanticEngine: Instance configurée
    """
    return SemanticEngine()


if __name__ == "__main__":
    # Test du moteur
    engine = SemanticEngine()
    
    # Données de test
    test_user_responses = {
        "q1_description": "I have experience in data cleaning with Python and creating dashboards with Tableau.",
        "q2_projects": "I worked on a machine learning project using regression models to predict sales.",
        "technologies": ["Python", "Tableau", "Scikit-learn"]
    }
    
    test_blocks = {
        "Data Analysis": ["data cleaning", "data visualization", "statistical analysis", "Python programming"],
        "Machine Learning": ["regression models", "classification", "model evaluation", "feature engineering"],
        "NLP": ["text preprocessing", "word embeddings", "sentiment analysis", "transformer models"]
    }
    
    # Analyse
    block_scores, overall = engine.analyze_responses(test_user_responses, test_blocks)
    
    print("=== Résultats du test ===")
    for block, score in block_scores.items():
        print(f"{block}: {score:.3f}")
    print(f"Score global: {overall:.3f}")
    print("=" * 30)
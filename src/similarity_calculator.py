"""

"""
import numpy as np
from typing import List, Dict, Tuple
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Scores aggregation method"""
    MEAN = "mean"
    MAX = "max"
    WEIGHTED_MEAN = "weighted_mean"
    QUANTILE = "quantile"


class SimilarityCalculator:
    """
    Compute the similarity with different algorithms
    
    Fonctionnalités :
    1. Plusieurs méthodes de similarité
    2. Méthodes d'agrégation configurables
    3. Seuils et normalisation
    """

    def __init__(self, method: str = "cosine"):
        self.method = method
    
    def compute_cosinus_similarity_matrix(
        self, 
        embeddings_a: np.ndarray, 
        embeddings_b: np.ndarray
    ) -> np.ndarray:
        """
        Compute the cosinus matrix similarity between two embeddings
        
        Args:
            embeddings_a: Matrice n x d
            embeddings_b: Matrice m x d
            
        Returns:
            np.ndarray: Matrice n x m de similarités
        """
        # Vectors normalisation
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        
        return np.dot(a_norm, b_norm.T)
    
    
    def aggregate_scores(
        self, 
        similarity_matrix: np.ndarray, 
        method: str = "weighted_mean",
        weights: np.ndarray = None,
    ) -> float:
        """
        Compute coverage score which is the aggregation of a the similarity matrix into an overall score
        
        Args:
            similarity_matrix: Matrice de similarité n x m
            method: Méthode d'agrégation
            weights: Poids pour l'agrégation pondérée
            
        Returns:
            float: Score agrégé
        """
        
        if weights is None:
            weights = np.ones(similarity_matrix.shape[1])
        # Moyenne pondérée par colonne
        weighted_scores = np.average(similarity_matrix, axis=1, weights=weights)
        return float(np.mean(weighted_scores))
        
        
    def compute_block_score(
        self,
        user_embeddings: np.ndarray,
        block_embeddings: np.ndarray,
        aggregation_method: AggregationMethod = AggregationMethod.MEAN
    ) -> float:
        """
        Calcule le score pour un bloc de compétences
        
        Args:
            user_embeddings: Embeddings des réponses utilisateur
            block_embeddings: Embeddings des compétences du bloc
            aggregation_method: Méthode d'agrégation
            
        Returns:
            float: Score du bloc
        """
        # Calculer la matrice de similarité
        similarity_matrix = self.compute_similarity_matrix(user_embeddings, block_embeddings)
        
        # Pour chaque réponse utilisateur, prendre le maximum avec les compétences
        max_per_user_response = np.max(similarity_matrix, axis=1)
        
        # Agrégation des scores max
        if aggregation_method == AggregationMethod.MEAN:
            return float(np.mean(max_per_user_response))
        elif aggregation_method == AggregationMethod.MAX:
            return float(np.max(max_per_user_response))
        elif aggregation_method == AggregationMethod.QUANTILE:
            return float(np.quantile(max_per_user_response, 0.75))
        else:
            return float(np.mean(max_per_user_response))
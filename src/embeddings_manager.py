"""
SimilarityCalculator - Calculs avancés de similarité
Responsable : Méthodes variées de calcul de similarité et agrégation
"""

import numpy as np
from typing import List, Dict, Tuple
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class SimilarityMethod(Enum):
    """Méthodes de calcul de similarité disponibles"""
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"

class AggregationMethod(Enum):
    """Méthodes d'agrégation des scores"""
    MEAN = "mean"
    MAX = "max"
    WEIGHTED_MEAN = "weighted_mean"
    QUANTILE = "quantile"

class SimilarityCalculator:
    """
    Calculateur de similarité avec différentes méthodes
    
    Fonctionnalités :
    1. Plusieurs méthodes de similarité
    2. Méthodes d'agrégation configurables
    3. Seuils et normalisation
    """
    
    def __init__(self, method: SimilarityMethod = SimilarityMethod.COSINE):
        self.method = method
    
    def compute_similarity_matrix(
        self, 
        embeddings_a: np.ndarray, 
        embeddings_b: np.ndarray
    ) -> np.ndarray:
        """
        Calcule la matrice de similarité entre deux ensembles d'embeddings
        
        Args:
            embeddings_a: Matrice n x d
            embeddings_b: Matrice m x d
            
        Returns:
            np.ndarray: Matrice n x m de similarités
        """
        if self.method == SimilarityMethod.COSINE:
            return self._cosine_similarity(embeddings_a, embeddings_b)
        elif self.method == SimilarityMethod.DOT_PRODUCT:
            return self._dot_product_similarity(embeddings_a, embeddings_b)
        elif self.method == SimilarityMethod.EUCLIDEAN:
            return self._euclidean_similarity(embeddings_a, embeddings_b)
        elif self.method == SimilarityMethod.MANHATTAN:
            return self._manhattan_similarity(embeddings_a, embeddings_b)
        else:
            raise ValueError(f"Méthode inconnue : {self.method}")
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Similarité cosinus"""
        # Normaliser les vecteurs
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        
        return np.dot(a_norm, b_norm.T)
    
    def _dot_product_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Produit scalaire"""
        return np.dot(a, b.T)
    
    def _euclidean_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Similarité basée sur la distance euclidienne"""
        # Distance euclidienne
        n = a.shape[0]
        m = b.shape[0]
        distances = np.zeros((n, m))
        
        for i in range(n):
            for j in range(m):
                distances[i, j] = np.linalg.norm(a[i] - b[j])
        
        # Convertir distance en similarité (plus la distance est petite, plus la similarité est grande)
        max_dist = np.max(distances)
        if max_dist > 0:
            similarities = 1 - (distances / max_dist)
        else:
            similarities = np.ones_like(distances)
        
        return similarities
    
    def _manhattan_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Similarité basée sur la distance de Manhattan"""
        n = a.shape[0]
        m = b.shape[0]
        distances = np.zeros((n, m))
        
        for i in range(n):
            for j in range(m):
                distances[i, j] = np.sum(np.abs(a[i] - b[j]))
        
        # Convertir distance en similarité
        max_dist = np.max(distances)
        if max_dist > 0:
            similarities = 1 - (distances / max_dist)
        else:
            similarities = np.ones_like(distances)
        
        return similarities
    
    def aggregate_scores(
        self, 
        similarity_matrix: np.ndarray, 
        method: AggregationMethod = AggregationMethod.MEAN,
        weights: np.ndarray = None,
        quantile: float = 0.75
    ) -> float:
        """
        Agrège une matrice de similarité en un score unique
        
        Args:
            similarity_matrix: Matrice de similarité n x m
            method: Méthode d'agrégation
            weights: Poids pour l'agrégation pondérée
            quantile: Quantile pour la méthode QUANTILE
            
        Returns:
            float: Score agrégé
        """
        if method == AggregationMethod.MEAN:
            return float(np.mean(similarity_matrix))
        
        elif method == AggregationMethod.MAX:
            return float(np.max(similarity_matrix))
        
        elif method == AggregationMethod.WEIGHTED_MEAN:
            if weights is None:
                weights = np.ones(similarity_matrix.shape[1])
            # Moyenne pondérée par colonne
            weighted_scores = np.average(similarity_matrix, axis=1, weights=weights)
            return float(np.mean(weighted_scores))
        
        elif method == AggregationMethod.QUANTILE:
            return float(np.quantile(similarity_matrix, quantile))
        
        else:
            raise ValueError(f"Méthode d'agrégation inconnue : {method}")
    
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
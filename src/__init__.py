"""
Package src - Modules pour le projet AISCA
"""

from .semantic_engine import SemanticEngine, create_semantic_engine
from .embeddings_manager import EmbeddingsManager
from .similarity_calculator import SimilarityCalculator, SimilarityMethod, AggregationMethod

__all__ = [
    'SemanticEngine',
    'create_semantic_engine',
    'EmbeddingsManager',
    'SimilarityCalculator',
    'SimilarityMethod',
    'AggregationMethod'
]
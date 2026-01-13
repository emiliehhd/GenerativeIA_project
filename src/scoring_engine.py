"""
ScoringEngine - Calcul avancé des scores par métier
Responsable : Formules de scoring et pondérations
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class ScoringEngine:
    """
    Moteur de calcul des scores d'adéquation métier
    """
    
    def __init__(self, weights_config: Optional[Dict] = None):
        """
        Args:
            weights_config: Configuration des poids {
                "skill_importance": {skill_id: weight},
                "block_weights": {block_name: weight},
                "experience_multiplier": float
            }
        """
        self.weights_config = weights_config or {}
        
    def calculate_job_score(
        self,
        job_id: str,
        semantic_scores: Dict[str, float],
        explicit_skills: List[str],
        job_requirements: Dict,
        user_experience: Optional[int] = None
    ) -> Dict:
        """
        Calcule un score complet pour un métier
        
        Returns:
            Dict avec score total et décomposition
        """
        # 1. Score sémantique (basé sur les blocs)
        semantic_score = self._calculate_semantic_score(
            job_requirements, semantic_scores
        )
        
        # 2. Score de match exact (compétences explicites)
        exact_match_score = self._calculate_exact_match_score(
            job_requirements, explicit_skills
        )
        
        # 3. Score combiné
        combined_score = self._combine_scores(
            semantic_score, exact_match_score
        )
        
        # 4. Ajustement par expérience
        final_score = self._apply_experience_adjustment(
            combined_score, user_experience
        )
        
        # 5. Décomposition pour explication
        score_breakdown = {
            "semantic": semantic_score,
            "exact_match": exact_match_score,
            "combined": combined_score,
            "experience_adjusted": final_score,
            "components": {
                "semantic_weight": 0.6,
                "exact_match_weight": 0.4
            }
        }
        
        return {
            "total": final_score,
            "breakdown": score_breakdown
        }
    
    def _calculate_semantic_score(
        self,
        job_requirements: Dict,
        semantic_scores: Dict[str, float]
    ) -> float:
        """
        Calcule le score basé sur la similarité sémantique
        """
        required_blocks = job_requirements.get("required_blocks", [])
        
        if not required_blocks:
            return 0.0
        
        block_scores = []
        block_weights = []
        
        for block_name in required_blocks:
            # Score du bloc
            block_score = semantic_scores.get(block_name, 0.0)
            
            # Poids du bloc (configurable)
            block_weight = self.weights_config.get(
                "block_weights", {}
            ).get(block_name, 1.0)
            
            block_scores.append(block_score)
            block_weights.append(block_weight)
        
        # Moyenne pondérée
        if sum(block_weights) > 0:
            semantic_score = np.average(block_scores, weights=block_weights)
        else:
            semantic_score = np.mean(block_scores) if block_scores else 0.0
        
        return float(semantic_score)
    
    def _calculate_exact_match_score(
        self,
        job_requirements: Dict,
        explicit_skills: List[str]
    ) -> float:
        """
        Calcule le score basé sur le match exact des compétences
        """
        required_skills = job_requirements.get("required_skills", [])
        
        if not required_skills or not explicit_skills:
            return 0.0
        
        # Compétences couvertes
        covered_skills = [
            skill for skill in required_skills 
            if skill in explicit_skills
        ]
        
        # Score basé sur la couverture
        coverage_ratio = len(covered_skills) / len(required_skills)
        
        # Ajustement par importance des compétences
        weighted_score = coverage_ratio
        
        if "skill_importance" in self.weights_config:
            # Calcul pondéré par importance
            total_weight = 0.0
            weighted_sum = 0.0
            
            for skill in required_skills:
                weight = self.weights_config["skill_importance"].get(skill, 1.0)
                total_weight += weight
                
                if skill in covered_skills:
                    weighted_sum += weight
            
            if total_weight > 0:
                weighted_score = weighted_sum / total_weight
        
        return float(weighted_score)
    
    def _combine_scores(
        self,
        semantic_score: float,
        exact_match_score: float
    ) -> float:
        """
        Combine les scores sémantique et exact
        """
        # Par défaut : 60% sémantique, 40% exact
        semantic_weight = 0.6
        exact_weight = 0.4
        
        combined = (
            semantic_weight * semantic_score +
            exact_weight * exact_match_score
        )
        
        return float(combined)
    
    def _apply_experience_adjustment(
        self,
        score: float,
        experience_years: Optional[int]
    ) -> float:
        """
        Ajuste le score basé sur l'expérience
        """
        if experience_years is None:
            return score
        
        # Multiplicateur basé sur l'expérience
        if experience_years >= 5:
            multiplier = 1.1  # +10% pour expérience avancée
        elif experience_years >= 2:
            multiplier = 1.05  # +5% pour expérience intermédiaire
        elif experience_years == 0:
            multiplier = 0.9  # -10% pour débutant
        else:
            multiplier = 1.0  # Pas d'ajustement
        
        adjusted = score * multiplier
        
        # Limiter à 1.0
        return min(adjusted, 1.0)
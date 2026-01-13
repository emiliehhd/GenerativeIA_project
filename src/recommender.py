"""
Recommender - Système principal de recommandation de métiers

"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass
from enum import Enum


## logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatchStrategy(Enum):
    """Job-skills matching strategy"""
    STRICT      = "strict"      # All skills required
    PARTIAL     = "partial"     # Score based on the coverage percentage
    WEIGHTED    = "weighted"    # Different weights per skills
    SEMANTIC    = "semantic"    # Use raw semantic score


@dataclass
class JobRecommendation:
    """Class to represent a job recommendation"""
    job_id: str
    title: str
    score: float                # Matching score (0-1)
    matched_skills: List[str]   # Compétences bien couvertes
    missing_skills: List[str]   # Skills to develop     
    explanation: str            # Textual explaination
    description: str            # Job description
    confidence: float           # Recommendation confidence score (0-1)
    
    def to_dict(self) -> Dict:
        """Convert to dictionnary for Streamlit"""
        return {
            "id": self.job_id,
            "title": self.title,
            "score": self.score,
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "explanation": self.explanation,
            "description": self.description,
            "confidence": self.confidence
        }

class RecommenderSystem:
    """
    Système de recommandation de métiers basé sur les compétences
    
    Fonctionnalities :
    1. Compute matching score for each job
    2. Semantic mathing for competencies
    3. Personalized recommendation
    4. Recommendation's explaination
    """
    
    def __init__(
        self,
        job_data: Dict,
        competency_data: Dict,
        strategy: MatchStrategy = MatchStrategy.SEMANTIC,
        min_score_threshold: float = 0.3
    ):
        """
        Initialize the recommendation system
        
        Args:
            job_data: Jobs' data (format structuré)
            competency_data: competencies' data
            strategy: Matching strategy
            min_score_threshold: Minimum threshold to recommand
        """
        self.job_data = job_data
        self.competency_data = competency_data
        self.strategy = strategy
        self.min_score_threshold = min_score_threshold
        
        #  Reversed Index for fast research
        self._build_indices()
        
        logger.info(f"RecommenderSystem initialized with stratégy: {strategy}")
    

    def _build_indices(self):
        """Build indices for fast research"""
        # Skill index → jobs
        self.skill_to_jobs = {}
        
        for job in self.job_data.get("jobs", []):
            job_id = job.get("id")
            required_skills = job.get("required_competencies", [])
            
            for skill_id in required_skills:
                if skill_id not in self.skill_to_jobs:
                    self.skill_to_jobs[skill_id] = []
                self.skill_to_jobs[skill_id].append(job_id)
        
        # Competencie index → data
        self.skill_data = {}
        for skill in self.competency_data.get("competencies", []):
            self.skill_data[skill["id"]] = skill
        
        logger.debug(f"Build index: {len(self.skill_to_jobs)} competencies → {len(self.job_data.get('jobs', []))} jobs")
    

    def recommend_jobs(
        self,
        semantic_scores: Dict[str, float],
        user_skills: Optional[List[str]] = None,
        top_k: int = 3,
        explain: bool = True
    ) -> List[JobRecommendation]:
        """
        Recommend jobs based on the semantic score
        
        Args:
            semantic_scores: Scores per block {block_name: score}
            user_skills: Explicit list of user's competencies
            top_k: Number of recommendations to return
            explain: Generate explaination
            
        Returns:
            List[JobRecommendation]: Top K recommandations
        """
        logger.info(f"Start recommendations (top_k={top_k})")
        
        # 1. Compute scores for each job
        job_scores = self._calculate_job_scores(semantic_scores, user_skills)
        
        # 2. Filter with minimum threshold
        filtered_jobs = {
            job_id: score 
            for job_id, score in job_scores.items() 
            if score >= self.min_score_threshold
        }
        
        if not filtered_jobs:
            logger.warning("No job reach the minimum threshold")
            return self._get_fallback_recommendations()
        
        # 3. Trier par score décroissant
        sorted_jobs = sorted(
            filtered_jobs.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 4. Prendre les top K
        top_jobs = sorted_jobs[:top_k]
        
        # 5. Construire les recommandations complètes
        recommendations = []
        for job_id, score in top_jobs:
            job = self._get_job_by_id(job_id)
            if not job:
                continue
                
            recommendation = self._build_job_recommendation(
                job, score, semantic_scores, user_skills, explain
            )
            recommendations.append(recommendation)
        
        logger.info(f"{len(recommendations)} recommandations générées")
        return recommendations


    def _calculate_job_scores(
        self,
        semantic_scores: Dict[str, float],
        user_skills: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calcule le score d'adéquation pour chaque métier
        
        Args:
            semantic_scores: Scores par bloc
            user_skills: Compétences explicites
            
        Returns:
            Dict[job_id: str, score: float]
        """
        job_scores = {}
        
        for job in self.job_data.get("jobs", []):
            job_id = job.get("id")
            
            if self.strategy == MatchStrategy.SEMANTIC:
                score = self._calculate_semantic_score(job, semantic_scores)
            elif self.strategy == MatchStrategy.PARTIAL:
                score = self._calculate_partial_score(job, user_skills)
            elif self.strategy == MatchStrategy.WEIGHTED:
                score = self._calculate_weighted_score(job, semantic_scores, user_skills)
            else:  # STRICT
                score = self._calculate_strict_score(job, user_skills)
            
            job_scores[job_id] = score
        
        return job_scores
    

    def _calculate_semantic_score(
        self,
        job: Dict,
        semantic_scores: Dict[str, float]
    ) -> float:
        """
        Calcule le score basé sur la similarité sémantique
        
        Logique : Un métier nécessite certains blocs de compétences.
        Le score du métier = moyenne des scores des blocs requis.
        """
        required_skills = job.get("required_competencies", [])
        
        if not required_skills:
            return 0.0
        
        # Pour chaque compétence requise, trouver son bloc
        block_scores_for_job = []
        
        for skill_id in required_skills:
            skill_info = self.skill_data.get(skill_id)
            if not skill_info:
                continue
                
            block_name = skill_info.get("block")
            if block_name in semantic_scores:
                block_scores_for_job.append(semantic_scores[block_name])
        
        if not block_scores_for_job:
            return 0.0
        
        # Moyenne des scores des blocs requis
        score = np.mean(block_scores_for_job)
        
        # Pénalité si trop de compétences manquantes
        missing_ratio = 1.0 - (len(block_scores_for_job) / len(required_skills))
        score *= (1.0 - 0.3 * missing_ratio)  # Pénalité de 30% max
        
        return float(score)
    

    def _calculate_partial_score(
        self,
        job: Dict,
        user_skills: Optional[List[str]] = None
    ) -> float:
        """
        Calcule le score basé sur le pourcentage de compétences couvertes
        """
        if not user_skills:
            return 0.0
        
        required_skills = job.get("required_competencies", [])
        
        if not required_skills:
            return 0.0
        
        # Compétences couvertes (match exact)
        covered = [skill for skill in required_skills if skill in user_skills]
        
        # Pourcentage de couverture
        coverage_ratio = len(covered) / len(required_skills)
        
        return float(coverage_ratio)
    

    def _calculate_weighted_score(
        self,
        job: Dict,
        semantic_scores: Dict[str, float],
        user_skills: Optional[List[str]] = None
    ) -> float:
        """
        Score pondéré : combine sémantique et match exact
        """
        semantic_part = self._calculate_semantic_score(job, semantic_scores)
        
        if not user_skills:
            return semantic_part
        
        exact_part = self._calculate_partial_score(job, user_skills)
        
        # Combinaison : 70% sémantique, 30% exact
        weighted_score = 0.7 * semantic_part + 0.3 * exact_part
        
        return float(weighted_score)
    

    def _calculate_strict_score(
        self,
        job: Dict,
        user_skills: Optional[List[str]] = None
    ) -> float:
        """
        Score strict : 1.0 si toutes les compétences requises sont couvertes
        """
        if not user_skills:
            return 0.0
        
        required_skills = job.get("required_competencies", [])
        
        # Vérifie si toutes les compétences requises sont présentes
        all_covered = all(skill in user_skills for skill in required_skills)
        
        return 1.0 if all_covered else 0.0
    

    def _build_job_recommendation(
        self,
        job: Dict,
        score: float,
        semantic_scores: Dict[str, float],
        user_skills: Optional[List[str]] = None,
        explain: bool = True
    ) -> JobRecommendation:
        """
        Construit une recommandation complète pour un métier
        """
        job_id = job.get("id")
        title = job.get("title", "Unknown Job")
        description = job.get("description", "")
        
        # Identifier les compétences couvertes et manquantes
        matched_skills, missing_skills = self._analyze_skill_coverage(
            job, semantic_scores, user_skills
        )
        
        # Générer une explication
        explanation = ""
        if explain:
            explanation = self._generate_explanation(
                title, score, matched_skills, missing_skills
            )
        
        # Calculer la confiance
        confidence = self._calculate_confidence(score, len(matched_skills), len(missing_skills))
        
        return JobRecommendation(
            job_id=job_id,
            title=title,
            score=score,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            explanation=explanation,
            description=description,
            confidence=confidence
        )
    

    def _analyze_skill_coverage(
        self,
        job: Dict,
        semantic_scores: Dict[str, float],
        user_skills: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Analyse quelles compétences sont couvertes et manquantes
        """
        required_skills = job.get("required_competencies", [])
        
        matched = []
        missing = []
        
        for skill_id in required_skills:
            skill_info = self.skill_data.get(skill_id)
            if not skill_info:
                continue
                
            skill_name = skill_info.get("text", skill_id)
            block_name = skill_info.get("block")
            
            # Vérifier la couverture
            is_covered = False
            
            # 1. Vérifier le score sémantique du bloc
            if block_name in semantic_scores:
                block_score = semantic_scores[block_name]
                if block_score >= 0.5:  # Seuil de couverture
                    is_covered = True
            
            # 2. Vérifier les compétences explicites
            if not is_covered and user_skills:
                if skill_id in user_skills or skill_name.lower() in [s.lower() for s in user_skills]:
                    is_covered = True
            
            if is_covered:
                matched.append(skill_name)
            else:
                missing.append(skill_name)
        
        return matched, missing
    

    def _generate_explanation(
        self,
        job_title: str,
        score: float,
        matched_skills: List[str],
        missing_skills: List[str]
    ) -> str:
        """
        Génère une explication textuelle de la recommandation
        """
        # Formatage du score en pourcentage
        score_percent = int(score * 100)
        
        explanation_parts = []
        
        # Partie 1 : Score global
        if score >= 0.8:
            explanation_parts.append(f"**Adéquation exceptionnelle ({score_percent}%)** avec le profil de {job_title}.")
        elif score >= 0.6:
            explanation_parts.append(f"**Bonne adéquation ({score_percent}%)** avec le profil de {job_title}.")
        elif score >= 0.4:
            explanation_parts.append(f"**Adéquation modérée ({score_percent}%)** avec {job_title}.")
        else:
            explanation_parts.append(f"**Adéquation limitée ({score_percent}%)** avec {job_title}.")
        
        # Partie 2 : Compétences fortes
        if matched_skills:
            if len(matched_skills) <= 3:
                skills_text = ", ".join(matched_skills)
                explanation_parts.append(f"**Points forts :** {skills_text}.")
            else:
                explanation_parts.append(f"**{len(matched_skills)} compétences clés** sont bien maîtrisées.")
        
        # Partie 3 : Compétences à développer
        if missing_skills:
            if len(missing_skills) <= 3:
                missing_text = ", ".join(missing_skills[:3])
                explanation_parts.append(f"**À développer :** {missing_text}.")
            else:
                explanation_parts.append(f"**{len(missing_skills)} compétences** nécessitent un développement.")
        
        # Partie 4 : Recommandation finale
        if score >= 0.7:
            explanation_parts.append("Ce métier correspond particulièrement bien à votre profil.")
        elif score >= 0.5:
            explanation_parts.append("Avec quelques formations complémentaires, ce métier pourrait vous correspondre.")
        else:
            explanation_parts.append("Ce métier nécessiterait des formations significatives.")
        
        return " ".join(explanation_parts)
    

    def _calculate_confidence(
        self,
        score: float,
        matched_count: int,
        missing_count: int
    ) -> float:
        """
        Calcule la confiance dans la recommandation
        
        Facteurs :
        1. Score d'adéquation (poids 50%)
        2. Ratio compétences couvertes (poids 30%)
        3. Nombre total de compétences évaluées (poids 20%)
        """
        # 1. Score d'adéquation
        score_factor = score
        
        # 2. Ratio de compétences couvertes
        total_skills = matched_count + missing_count
        if total_skills > 0:
            coverage_ratio = matched_count / total_skills
        else:
            coverage_ratio = 0.0
        
        # 3. Facteur de complétude (plus on a évalué de compétences, plus c'est fiable)
        completeness_factor = min(total_skills / 10, 1.0)  # Normalisé sur 10 compétences
        
        # Combinaison pondérée
        confidence = (
            0.5 * score_factor +
            0.3 * coverage_ratio +
            0.2 * completeness_factor
        )
        
        return float(confidence)
    

    def _get_job_by_id(self, job_id: str) -> Optional[Dict]:
        """Retrouve un métier par son ID"""
        for job in self.job_data.get("jobs", []):
            if job.get("id") == job_id:
                return job
        return None
    
    
    def _get_fallback_recommendations(self) -> List[JobRecommendation]:
        """
        Recommandations de fallback quand aucun métier ne passe le seuil
        """
        logger.info("Utilisation des recommandations de fallback")
        
        fallback_jobs = [
            {
                "id": "FB001",
                "title": "Data Analyst Junior",
                "description": "Profil d'entrée en analyse de données nécessitant des bases en statistiques et visualisation.",
                "required_competencies": ["C01", "C02"]  # Compétences de base
            },
            {
                "id": "FB002", 
                "title": "Technicien Informatique",
                "description": "Poste généraliste en informatique avec possibilité de spécialisation.",
                "required_competencies": ["C03"]  # Programmation de base
            }
        ]
        
        recommendations = []
        for job in fallback_jobs:
            recommendations.append(JobRecommendation(
                job_id=job["id"],
                title=job["title"],
                score=0.4,  # Score de fallback
                matched_skills=["Compétences de base"],
                missing_skills=["Spécialisations techniques"],
                explanation="Recommandation de base pour explorer le domaine.",
                description=job["description"],
                confidence=0.3
            ))
        
        return recommendations
    
    def get_recommendation_stats(self) -> Dict:
        """Retourne des statistiques sur le système"""
        return {
            "total_jobs": len(self.job_data.get("jobs", [])),
            "total_skills": len(self.skill_data),
            "strategy": self.strategy.value,
            "min_threshold": self.min_score_threshold,
            "skill_to_jobs_mapping_size": len(self.skill_to_jobs)
        }


# Factory function pour créer le système
def create_recommender(
    job_data_path: str = "data/jobs.json",
    competency_data_path: str = "data/competencies.json"
) -> RecommenderSystem:
    """
    Crée une instance du système de recommandation
    
    Args:
        job_data_path: Chemin vers les données métiers
        competency_data_path: Chemin vers les données compétences
        
    Returns:
        RecommenderSystem: Instance configurée
    """
    import json
    
    with open(job_data_path, 'r', encoding='utf-8') as f:
        job_data = json.load(f)
    
    with open(competency_data_path, 'r', encoding='utf-8') as f:
        competency_data = json.load(f)
    
    return RecommenderSystem(job_data, competency_data)
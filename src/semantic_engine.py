"""
SemanticEngine - Moteur d'analyse sémantique avec SBERT
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List, Tuple, Any
import logging
import os

from weight_matrix import JOBS, DOMAINS, MATRIX, WeightMatrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticEngine:
    """
    Moteur d'analyse sémantique pour comparer les réponses utilisateur
    avec le référentiel de compétences
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialise le moteur sémantique
        
        Args:
            model_name: Nom du modèle SBERT à utiliser
        """
        logger.info(f"Chargement du modèle SBERT: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.competences_df = None
        self.metiers_df = None
        self.competence_embeddings = None
        

    def load_data(self, competences_path: str, metiers_path: str):
        """
        Charge les données des référentiels
        
        Args:
            competences_path: Chemin vers le fichier des compétences
            metiers_path: Chemin vers le fichier des métiers
        """
        # Charger les compétences
        self.competences_df = pd.read_csv(competences_path)
        logger.info(f"Compétences chargées: {len(self.competences_df)}")
        
        # Charger les métiers
        self.metiers_df = pd.read_csv(metiers_path)
        logger.info(f"Métiers chargés: {len(self.metiers_df)}")
        
        # Pré-calculer les embeddings des compétences
        self._precompute_embeddings()
    

    def _precompute_embeddings(self):
        """Pré-calcul les embeddings des compétences pour optimisation"""
        if self.competences_df is not None:
            competence_texts = self.competences_df['competence'] + " - " + self.competences_df['description']
            self.competence_embeddings = self.model.encode(
                competence_texts.tolist(),
                convert_to_tensor=True,
                show_progress_bar=True
            )
            logger.info(f"Embeddings pré-calculés pour {len(competence_texts)} compétences")
    

    def analyze_user_profile(self, user_responses: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyse le profil utilisateur basé sur ses réponses
        
        Args:
            user_responses: Dictionnaire avec les réponses utilisateur
                - competences: texte des compétences
                - experiences: texte des expériences
                - outils: texte des outils maîtrisés
        
        Returns:
            Dict: Résultats de l'analyse
        """
        logger.info("Analyse du profil utilisateur")
        
        # Extraire et combiner les textes utilisateur
        user_texts = []
        for key in ['competences', 'experiences', 'outils']:

            if key in user_responses and user_responses[key]:
                user_texts.append(user_responses[key])
        
        if not user_texts:
            logger.warning("Aucune réponse utilisateur fournie")
            return {}
        
        # Encoder les réponses utilisateur
        user_embeddings = self.model.encode(user_texts, convert_to_tensor=True)
        
        # Calculer les similarités avec les compétences
        similarity_matrix = util.cos_sim(user_embeddings, self.competence_embeddings)
        
        # Pour chaque compétence, prendre la similarité maximale
        max_similarities = similarity_matrix.max(dim=0).values.cpu().numpy()
        
        # Associer les scores aux compétences
        self.competences_df['similarity_score'] = max_similarities

        # Dans semantic_engine.py, après ligne 105 :
        print("=== SCORES SÉMANTIQUES BRUTS ===")
        for i, score in enumerate(max_similarities):
            comp_name = self.competences_df.iloc[i]['competence']
            print(f"{comp_name}: {score:.4f}")
        print("="*40)
        
        # Calculer les scores par bloc
        bloc_scores = self._calculate_bloc_scores()

        
        # Calculer le score global avec matrice de poids
        metier_scores = []
        for job_id in JOBS.keys():
            score = WeightMatrix.get_job_score(bloc_scores, job_id)
            metier_scores.append(score)
        
        #ICI
        # Recommend jobs
        # metier_recommendations = self._recommend_metiers(bloc_scores)
        metier_recommendations = self._recommend_metiers_with_weights(bloc_scores)  
        
        # Identifier les compétences fortes et faibles
        strong_competences = self._identify_strong_competences()
        weak_competences = self._identify_weak_competences()
        
        # Score global
        overall_score = np.mean(list(bloc_scores.values()))

        return {
            'bloc_scores': bloc_scores,
            'overall_score': float(overall_score),
            'metier_recommendations': metier_recommendations,
            'strong_competences': strong_competences,
            'weak_competences': weak_competences,
            'competence_details': self.competences_df[['competence', 'bloc', 'similarity_score']].to_dict('records')
        }


    def _calculate_bloc_scores(self) -> Dict[str, float]:
        """Calcule le score moyen par bloc de compétences"""
        if self.competences_df is None or 'similarity_score' not in self.competences_df.columns:
            return {}
        
        bloc_scores = {}
        for bloc in self.competences_df['bloc'].unique():
            bloc_competences = self.competences_df[self.competences_df['bloc'] == bloc]
            bloc_score = bloc_competences['similarity_score'].mean()
            bloc_scores[bloc] = float(bloc_score)
        
        return bloc_scores
    

    def _recommend_metiers_with_weights(self, bloc_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Recommande des métiers avec la matrice de poids
        
        Args:
            bloc_scores: Scores par bloc
        
        Returns:
            List: Métiers recommandés
        """
        # Utiliser la matrice de poids pour les recommandations
        recommendations = WeightMatrix.get_recommendations(bloc_scores, top_k=3)
        
        # Formater les résultats
        formatted_recommendations = []
        for rec in recommendations:
            formatted_recommendations.append({
                'metier': rec['name'],
                'score': rec['score'],
                'id': rec['id'],
                'explanation': WeightMatrix.explain_score(rec['id'], bloc_scores),
                'description': self._get_job_description(rec['id'])
            })
        
        logger.info(f"Recommandations avec poids: {[(r['metier'], r['score']) for r in formatted_recommendations]}")
        
        return formatted_recommendations

    #ICI
    def _get_job_description(self, job_id: str) -> str:
        """Récupère la description d'un métier depuis le CSV"""
        if self.metiers_df is not None:
            job_row = self.metiers_df[self.metiers_df['metier_id'] == job_id]
            if not job_row.empty:
                return job_row.iloc[0].get('description', '')
        return "Description non disponible"

    
    def _identify_strong_competences(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Identifie les compétences bien maîtrisées"""
        strong = self.competences_df[self.competences_df['similarity_score'] >= threshold]
        return strong[['competence', 'bloc', 'similarity_score']].to_dict('records')
    
    def _identify_weak_competences(self, threshold: float = 0.4) -> List[Dict[str, Any]]:
        """Identifie les compétences à améliorer"""
        weak = self.competences_df[self.competences_df['similarity_score'] < threshold]
        return weak[['competence', 'bloc', 'similarity_score']].to_dict('records')

    
    def calculate_weighted_score_for_job(self, bloc_scores: Dict[str, float], job_id: str) -> Dict[str, Any]:
        """
        Calcule le score détaillé pour un métier spécifique
        
        Args:
            bloc_scores: Scores par bloc
            job_id: ID du métier
        
        Returns:
            Dict: Score détaillé avec explications
        """
        if job_id not in WeightMatrix.MATRIX:
            return {"score": 0.0, "details": []}
        
        job_weights = WeightMatrix.MATRIX[job_id]
        details = []
        total_weighted = 0.0
        total_weight = 0.0
        
        for domain_id, weight in job_weights.items():
            if domain_id in WeightMatrix.DOMAINS:
                domain_name = WeightMatrix.DOMAINS[domain_id]
                bloc_score = WeightMatrix._find_bloc_score(bloc_scores, domain_name)
                
                weighted_score = bloc_score * weight
                total_weighted += weighted_score
                total_weight += weight
                
                details.append({
                    'domain': domain_name,
                    'bloc_score': bloc_score,
                    'weight': weight,
                    'weighted_score': weighted_score,
                    'importance': 'Haute' if weight >= 0.8 else 'Moyenne' if weight >= 0.4 else 'Basse'
                })
        
        final_score = total_weighted / total_weight if total_weight > 0 else 0.0
        
        return {
            'score': final_score,
            'details': details,
            'job_name': WeightMatrix.JOBS.get(job_id, "Inconnu")
        }
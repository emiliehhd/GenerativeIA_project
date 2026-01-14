"""
SemanticEngine - Moteur d'analyse sémantique avec SBERT
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List, Tuple, Any
import logging
import os

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
                - auto-evaluation des competences : likert 1 a 5
                - experiences: texte des expériences
                - outils: texte des outils maîtrisés
        
        Returns:
            Dict: Résultats de l'analyse
        """
        logger.info("Analyse du profil utilisateur")
        
        # Extraire et combiner les textes utilisateur
        user_texts = []
        for key in ['competences', 'experiences', 'outils', 'python_score', 'sql_score']:
            if key in user_responses and user_responses[key]:
                print(">>>", key, "\n",user_responses[key])
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

        # # Ajuster les scores basés sur l'échelle de Likert
        # self._adjust_scores_with_likert(user_responses) #ICI
        
        # Calculer les scores par bloc
        bloc_scores = self._calculate_bloc_scores()
        
        # Recommander des métiers
        metier_recommendations = self._recommend_metiers(bloc_scores)
        
        # Identifier les compétences fortes et faibles
        strong_competences = self._identify_strong_competences()
        weak_competences = self._identify_weak_competences()
        
        # Score global
        overall_score = np.mean(list(bloc_scores.values()))

        # # Stocker les scores Likert dans les résultats #ICI
        # likert_scores = {}
        # if 'python_level' in user_responses:
        #     likert_scores['python'] = user_responses['python_level']
        # if 'sql_level' in user_responses:
        #     likert_scores['sql'] = user_responses['sql_level']
        
        return {
            'bloc_scores': bloc_scores,
            'overall_score': float(overall_score),
            # 'likert_scores': likert_scores,
            'metier_recommendations': metier_recommendations,
            'strong_competences': strong_competences,
            'weak_competences': weak_competences,
            'competence_details': self.competences_df[['competence', 'bloc', 'similarity_score']].to_dict('records')
        }
    
    
# def _adjust_scores_with_likert(self, user_responses: Dict[str, Any]):
#     """
#     Ajuste les scores sémantiques basés sur les réponses Likert
#     """
#     logger.info(f"Ajustement avec scores Likert: {user_responses.get('python_level')}, {user_responses.get('sql_level')}")
    
#     # Vérifier si les scores Likert existent
#     has_python_likert = 'python_level' in user_responses
#     has_sql_likert = 'sql_level' in user_responses
    
#     if not (has_python_likert or has_sql_likert):
#         logger.info("Aucun score Likert fourni, pas d'ajustement")
#         return
    
#     # Ajuster chaque compétence
#     for competence_idx, row in self.competences_df.iterrows():
#         competence_name = row['competence'].lower()
#         current_score = row['similarity_score']
        
#         # Vérifier Python
#         if has_python_likert and ('python' in competence_name):
#             python_score = user_responses['python_level']
#             # Convertir 1-5 → 0-1
#             likert_normalized = (python_score - 1) / 4.0
            
#             # Log pour debug
#             logger.debug(f"Python: {competence_name} - Score sémantique: {current_score:.3f}, Likert: {python_score}->{likert_normalized:.3f}")
            
#             # Combinaison : 60% sémantique, 40% Likert
#             adjusted_score = (0.6 * current_score) + (0.4 * likert_normalized)
#             self.competences_df.at[competence_idx, 'similarity_score'] = adjusted_score
            
#             logger.debug(f"  → Ajusté à: {adjusted_score:.3f}")
        
#         # Vérifier SQL
#         elif has_sql_likert and ('sql' in competence_name or 'query' in competence_name):
#             sql_score = user_responses['sql_level']
#             likert_normalized = (sql_score - 1) / 4.0
            
#             logger.debug(f"SQL: {competence_name} - Score sémantique: {current_score:.3f}, Likert: {sql_score}->{likert_normalized:.3f}")
            
#             adjusted_score = (0.6 * current_score) + (0.4 * likert_normalized)
#             self.competences_df.at[competence_idx, 'similarity_score'] = adjusted_score
            
#             logger.debug(f"  → Ajusté à: {adjusted_score:.3f}")
    
#     logger.info("Ajustement Likert terminé")


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
    
    def _recommend_metiers(self, bloc_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Recommande des métiers basés sur les scores de blocs"""
        if self.metiers_df is None:
            return []
        
        recommendations = []
        
        for _, metier_row in self.metiers_df.iterrows():
            metier_score = 0.0
            competence_ids = str(metier_row['competences_requises']).split(';')
            
            # Calculer le score pour ce métier
            for comp_id in competence_ids:
                comp_row = self.competences_df[self.competences_df['competence_id'] == comp_id.strip()]
                if not comp_row.empty:
                    bloc = comp_row.iloc[0]['bloc']
                    if bloc in bloc_scores:
                        metier_score += bloc_scores[bloc]
            
            # Normaliser le score
            if competence_ids and competence_ids[0].strip():
                metier_score /= len(competence_ids)
            
            recommendations.append({
                'metier': metier_row['metier'],
                'score': float(metier_score),
                'description': metier_row['description']
            })
        
        # Trier par score décroissant
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:3]  # Retourner les 3 meilleurs
    
    def _identify_strong_competences(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Identifie les compétences bien maîtrisées"""
        strong = self.competences_df[self.competences_df['similarity_score'] >= threshold]
        return strong[['competence', 'bloc', 'similarity_score']].to_dict('records')
    
    def _identify_weak_competences(self, threshold: float = 0.4) -> List[Dict[str, Any]]:
        """Identifie les compétences à améliorer"""
        weak = self.competences_df[self.competences_df['similarity_score'] < threshold]
        return weak[['competence', 'bloc', 'similarity_score']].to_dict('records')
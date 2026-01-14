"""
GenAIHandler - Gestion des appels à l'API Gemini avec cache
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenAIHandler:
    """
    Handler pour les appels API GenAI avec système de cache
    """
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "cache"):
        """
        Initialise le handler GenAI
        
        Args:
            api_key: Clé API Gemini (optionnelle, peut être dans .env)
            cache_dir: Dossier pour le cache
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialiser l'API (mode simulation si pas de clé)
        self.api_available = False
        if api_key or os.getenv('GEMINI_API_KEY'):
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key or os.getenv('GEMINI_API_KEY'))
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                self.api_available = True
                logger.info("API Gemini configurée")
            except ImportError:
                logger.warning("google-generativeai non installé, mode simulation")
        else:
            logger.warning("Pas de clé API fournie, mode simulation")
    
    def generate_progression_plan(self, analysis_results: Dict[str, Any]) -> str:
        """
        Génère un plan de progression personnalisé
        
        Args:
            analysis_results: Résultats de l'analyse sémantique
        
        Returns:
            str: Plan de progression formaté
        """
        cache_key = self._generate_cache_key("plan", analysis_results)
        
        # Vérifier le cache
        cached = self._get_from_cache(cache_key)
        if cached:
            logger.info("Plan récupéré du cache")
            return cached
        
        # Préparer les données pour le prompt
        weak_competences = analysis_results.get('weak_competences', [])
        recommendations = analysis_results.get('metier_recommendations', [])
        
        if recommendations:
            target_job = recommendations[0]['metier']
        else:
            target_job = "Data Analyst"
        
        # Générer le plan
        if self.api_available and len(weak_competences) > 0:
            try:
                prompt = self._build_progression_prompt(weak_competences, target_job)
                response = self.model.generate_content(prompt)
                plan = response.text
                
                # Mettre en cache
                self._save_to_cache(cache_key, plan)
                return plan
                
            except Exception as e:
                logger.error(f"API Error: {e}")
                return self._generate_default_plan(weak_competences, target_job)
        else:
            return self._generate_default_plan(weak_competences, target_job)
    

    def generate_professional_bio(self, user_info: Dict[str, Any], 
                                 analysis_results: Dict[str, Any]) -> str:
        """
        Génère une bio professionnelle
        
        Args:
            user_info: Informations utilisateur
            analysis_results: Résultats de l'analyse
        
        Returns:
            str: Bio professionnelle
        """
        cache_key = self._generate_cache_key("bio", {**user_info, **analysis_results})
        
        # Vérifier le cache
        cached = self._get_from_cache(cache_key)
        if cached:
            logger.info("Bio récupérée du cache")
            return cached
        
        # Préparer les données
        strong_competences = analysis_results.get('strong_competences', [])
        recommendations = analysis_results.get('metier_recommendations', [])
        
        if recommendations:
            target_job = recommendations[0]['metier']
            job_score = recommendations[0]['score']
        else:
            target_job = "Data Professional"
            job_score = 0.5
        
        # Générer la bio
        if self.api_available:
            try:
                prompt = self._build_bio_prompt(user_info, strong_competences, target_job, job_score)
                response = self.model.generate_content(prompt)
                bio = response.text
                
                # Mettre en cache
                self._save_to_cache(cache_key, bio)
                return bio
                
            except Exception as e:
                logger.error(f"API error: {e}")
                return self._generate_default_bio(user_info, strong_competences, target_job)
        else:
            return self._generate_default_bio(user_info, strong_competences, target_job)
    

    def _build_progression_prompt(self, weak_competences: list, target_job: str) -> str:
        """Construit le prompt pour le plan de progression"""
        competences_text = "\n".join([f"- {comp['competence']} ({comp['bloc']})" 
                                     for comp in weak_competences[:5]])
        
        return f"""
        You are a career counselor specializing in data-related professions.
        The user is aiming for the position of {target_job} but has gaps in the following areas:
        
        {competences_text}
        
        Generate a personalized and concrete development plan with:
        1. 3, 6, and 12-month objectives
        2. Specific resources to recommend (courses, books, projects)
        3. Practical projects to complete
        4. Progress indicators
        
        Be specific, realistic, and motivating. Respond in English.
        """
    
    def _build_bio_prompt(self, user_info: dict, strong_competences: list, 
                         target_job: str, job_score: float) -> str:
        """Construit le prompt pour la bio professionnelle"""
        competences_text = ", ".join([comp['competence'] for comp in strong_competences[:3]])
        name = user_info.get('name', 'le candidat')
        
        return f"""
        Tu es un expert en recrutement tech. Rédige une bio professionnelle percutante pour LinkedIn.
        
        Informations:
        - Nom: {name}
        - Poste visé: {target_job}
        - Adéquation avec le poste: {job_score:.0%}
        - Compétences principales: {competences_text}
        
        Exigences:
        - Style professionnel LinkedIn
        - 100-120 mots maximum
        - Mettre en avant les réalisations potentielles
        - Inclure des verbes d'action
        - Orienter vers les résultats
        
        Format:
        1. Phrase d'accroche
        2. Compétences et expertise
        3. Valeur ajoutée
        4. Objectif professionnel
        
        Réponds en anglais.
        """
    
    def _generate_default_plan(self, weak_competences: list, target_job: str) -> str:
        """Génère un plan par défaut (sans API)"""
        competences_text = ", ".join([comp['competence'] for comp in weak_competences[:3]])
        
        return f"""
        PLAN DE PROGRESSION POUR {target_job.upper()}
        
        Compétences à développer: {competences_text}
        
        """
    
    def _generate_default_bio(self, user_info: dict, strong_competences: list, 
                            target_job: str) -> str:
        """Génère une bio par défaut (sans API)"""
        name = user_info.get('name', '')
        competences_text = ", ".join([comp['competence'] for comp in strong_competences[:3]])
        
        return f"""
        {name if name else 'Professionnel'} avec {experience} ans d'expérience dans le domaine des données.
        
        Expertise en {competences_text if competences_text else 'analyse de données'}.
        Capacité à transformer des données complexes en insights actionnables.
        Expérience dans la mise en œuvre de solutions data-driven.
        
        À la recherche d'opportunités en tant que {target_job} pour contribuer à des projets innovants.
        """
    
    def _generate_cache_key(self, prefix: str, data: dict) -> str:
        """Génère une clé de cache unique"""
        data_str = json.dumps(data, sort_keys=True)
        data_hash = hashlib.md5(data_str.encode()).hexdigest()[:10]
        return f"{prefix}_{data_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """Récupère du cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Vérifier l'âge du cache (7 jours max)
                cache_date = datetime.fromisoformat(data.get('timestamp', ''))
                if (datetime.now() - cache_date).days <= 7:
                    return data.get('content')
                else:
                    os.remove(cache_file)
            except:
                pass
        
        return None
    
    def _save_to_cache(self, cache_key: str, content: str):
        """Sauvegarde dans le cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        cache_data = {
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
        except:
            pass
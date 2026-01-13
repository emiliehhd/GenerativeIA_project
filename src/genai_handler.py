"""
GenAIHandler - Gestion des appels à Gemini 2.5 Flash avec cache
Respect des contraintes : 1 appel pour le plan, 1 appel pour la bio
"""

import google.generativeai as genai
import hashlib
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class GenAIHandler:
    """
    Handler pour Google Gemini 2.5 Flash avec cache automatique
    """
    
    def __init__(self, api_key: str = None, cache_dir: str = "cache/genai"):
        """
        Args:
            api_key: Clé API Gemini (ou variable d'environnement GEMINI_API_KEY)
            cache_dir: Dossier pour le cache
        """
        # 1. Configuration API
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Clé API Gemini non trouvée. Définissez GEMINI_API_KEY")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # 2. Configuration cache
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 3. Compteur d'appels (pour monitoring)
        self.call_count = {
            "plan_progression": 0,
            "bio_professionnelle": 0,
            "cache_hits": 0
        }
        
        logger.info("GenAIHandler initialisé avec Gemini 2.5 Flash")
    
    def generate_progression_plan(self, context: Dict) -> str:
        """
        Génère un plan de progression personnalisé - 1 SEUL APPEL API
        
        Args:
            context: {
                "user_profile": dict,
                "block_scores": dict,
                "recommended_jobs": list,
                "weak_blocks": dict
            }
        
        Returns:
            str: Plan de progression formaté
        """
        cache_key = self._generate_cache_key("plan", context)
        
        # Vérifier le cache
        cached = self._get_from_cache(cache_key)
        if cached:
            self.call_count["cache_hits"] += 1
            logger.info("Plan de progression récupéré du cache")
            return cached
        
        # Si pas en cache, faire l'appel API
        self.call_count["plan_progression"] += 1
        
        # Construire le prompt
        prompt = self._build_progression_prompt(context)
        
        try:
            # UN SEUL APPEL API comme exigé
            response = self.model.generate_content(prompt)
            
            # Extraire le texte
            plan_text = response.text.strip()
            
            # Mettre en cache
            self._save_to_cache(cache_key, plan_text)
            
            logger.info(f"Plan de progression généré (appel #{self.call_count['plan_progression']})")
            return plan_text
            
        except Exception as e:
            logger.error(f"Erreur génération plan: {e}")
            return self._get_fallback_plan(context)
    
    def generate_professional_bio(self, context: Dict) -> str:
        """
        Génère une bio professionnelle - 1 SEUL APPEL API
        
        Args:
            context: {
                "user_profile": dict,
                "block_scores": dict,
                "recommended_jobs": list,
                "experience_years": int
            }
        
        Returns:
            str: Bio professionnelle formatée
        """
        cache_key = self._generate_cache_key("bio", context)
        
        # Vérifier le cache
        cached = self._get_from_cache(cache_key)
        if cached:
            self.call_count["cache_hits"] += 1
            logger.info("Bio récupérée du cache")
            return cached
        
        # Si pas en cache, faire l'appel API
        self.call_count["bio_professionnelle"] += 1
        
        # Construire le prompt
        prompt = self._build_bio_prompt(context)
        
        try:
            # UN SEUL APPEL API comme exigé
            response = self.model.generate_content(prompt)
            
            # Extraire le texte
            bio_text = response.text.strip()
            
            # Mettre en cache
            self._save_to_cache(cache_key, bio_text)
            
            logger.info(f"Bio générée (appel #{self.call_count['bio_professionnelle']})")
            return bio_text
            
        except Exception as e:
            logger.error(f"Erreur génération bio: {e}")
            return self._get_fallback_bio(context)
    
    def _build_progression_prompt(self, context: Dict) -> str:
        """Construit le prompt pour le plan de progression"""
        user = context.get("user_profile", {})
        scores = context.get("block_scores", {})
        weak = context.get("weak_blocks", {})
        jobs = context.get("recommended_jobs", [])
        
        prompt = f"""
        Tu es un conseiller en orientation professionnelle spécialisé dans les métiers de la data et de l'IA.
        
        PROFIL UTILISATEUR:
        - Nom: {user.get('name', 'Utilisateur')}
        - Expérience: {user.get('experience_years', 0)} ans
        - Métier recommandé: {jobs[0]['title'] if jobs else 'Data Analyst'}
        
        COMPÉTENCES ANALYSÉES (score 0-1):
        {self._format_scores_for_prompt(scores)}
        
        POINTS À DÉVELOPPER (scores < 0.5):
        {self._format_weak_points(weak)}
        
        TÂCHE:
        Génère un plan de progression PERSONNALISÉ et ACTIONNABLE pour combler les lacunes.
        
        FORMAT REQUIS:
        1. **Objectif professionnel** (1 phrase)
        2. **Timeline recommandée** (3-6-12 mois)
        3. **Ressources concrètes** (cours, projets, certifications)
        4. **Projets pratiques** à réaliser
        5. **Indicateurs de succès**
        
        Sois concret, réaliste et motivant. Maximum 300 mots.
        """
        
        return prompt.strip()
    
    def _build_bio_prompt(self, context: Dict) -> str:
        """Construit le prompt pour la bio professionnelle"""
        user = context.get("user_profile", {})
        scores = context.get("block_scores", {})
        jobs = context.get("recommended_jobs", [])
        tech = context.get("technologies", [])
        
        # Identifier les compétences fortes
        strong_skills = [k for k, v in scores.items() if v >= 0.6]
        
        prompt = f"""
        Tu es un expert en recrutement tech. Rédige une bio professionnelle PERSUASIVE.
        
        CONTEXTE:
        - Nom: {user.get('name', 'Candidat')}
        - Expérience: {user.get('experience_years', 0)} ans
        - Poste visé: {jobs[0]['title'] if jobs else 'Data Professional'}
        
        COMPÉTENCES PRINCIPALES:
        {', '.join(strong_skills[:3]) if strong_skills else 'Analyse de données'}
        
        TECHNOLOGIES: {', '.join(tech[:5]) if tech else 'Python, SQL'}
        
        CONSIGNES:
        1. Style professionnel LinkedIn
        2. 80-100 mots maximum
        3. Highlight des réalisations potentielles
        4. Inclure des verbes d'action
        5. Orienté résultats
        
        FORMAT:
        - Phrase d'accroche percutante
        - Compétences clés
        - Valeur ajoutée
        - Objectif professionnel
        
        Écris en français.
        """
        
        return prompt.strip()
    
    def _generate_cache_key(self, prefix: str, context: Dict) -> str:
        """Génère une clé de cache unique basée sur le contexte"""
        # Créer une chaîne stable du contexte
        context_str = json.dumps(context, sort_keys=True)
        
        # Hash MD5 pour une clé fixe
        hash_obj = hashlib.md5(context_str.encode())
        context_hash = hash_obj.hexdigest()[:8]
        
        return f"{prefix}_{context_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """Récupère une réponse du cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Vérifier la date (cache valide 7 jours)
                cache_date = datetime.fromisoformat(data.get("timestamp", ""))
                age_days = (datetime.now() - cache_date).days
                
                if age_days <= 7:  # Cache valide 1 semaine
                    return data.get("response")
                else:
                    os.remove(cache_file)  # Supprimer cache expiré
                    
            except Exception as e:
                logger.warning(f"Erreur lecture cache: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, response: str):
        """Sauvegarde une réponse dans le cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        cache_data = {
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "model": "gemini-1.5-flash"
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Réponse mise en cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Erreur sauvegarde cache: {e}")
    
    def _format_scores_for_prompt(self, scores: Dict) -> str:
        """Formate les scores pour le prompt"""
        return "\n".join([f"- {k}: {v:.1%}" for k, v in scores.items()])
    
    def _format_weak_points(self, weak: Dict) -> str:
        """Formate les points faibles pour le prompt"""
        if not weak:
            return "Aucun point faible majeur identifié."
        return "\n".join([f"- {k} ({v:.1%})" for k, v in weak.items()])
    
    def _get_fallback_plan(self, context: Dict) -> str:
        """Plan de secours si API échoue"""
        weak = context.get("weak_blocks", {})
        
        plan = f"""
        **Plan de développement professionnel**
        
        **Objectif** : Renforcer vos compétences pour le poste de {context.get('recommended_jobs', [{}])[0].get('title', 'Data Professional')}
        
        **Actions prioritaires (3 mois)** :
        1. Suivre un cours en ligne sur les domaines identifiés
        2. Pratiquer sur des projets concrets (Kaggle, GitHub)
        3. Rejoindre des communautés professionnelles
        
        **Compétences à développer** :
        {', '.join(list(weak.keys())[:3]) if weak else 'Aucune compétence faible majeure'}
        
        **Ressources recommandées** :
        - Coursera/edX pour les bases
        - Documentation officielle des technologies
        - Projets open-source pour la pratique
        """
        
        return plan.strip()
    
    def _get_fallback_bio(self, context: Dict) -> str:
        """Bio de secours si API échoue"""
        user = context.get("user_profile", {})
        exp = user.get("experience_years", 0)
        
        bio = f"""
        Professionnel avec {exp} ans d'expérience, spécialisé dans l'analyse de données et les technologies émergentes.
        
        Expertise dans l'extraction d'insights à partir de données complexes et la création de rapports décisionnels.
        Compétences techniques solides complétées par une forte capacité d'analyse et de résolution de problèmes.
        
        À la recherche d'opportunités pour contribuer à des projets innovants dans le domaine de la data.
        """
        
        return bio.strip()
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques d'utilisation"""
        return {
            **self.call_count,
            "total_api_calls": self.call_count["plan_progression"] + self.call_count["bio_professionnelle"],
            "cache_hit_ratio": f"{self.call_count['cache_hits'] / max(1, self.call_count['cache_hits'] + self.call_count['plan_progression'] + self.call_count['bio_professionnelle']):.1%}"
        }
    
    def clear_cache(self):
        """Vide le cache"""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir)
            logger.info("Cache vidé")
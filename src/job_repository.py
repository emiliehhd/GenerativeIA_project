"""
JobRepository - Gestion des données métiers
Responsable : Chargement, validation et accès aux données métiers
"""

import json
import yaml
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class JobRepository:
    """
    Repository pour la gestion des données métiers
    """
    
    def __init__(self, data_path: str = "data/"):
        self.data_path = Path(data_path)
        self.jobs = []
        self.job_categories = {}
        self.skill_relationships = {}
        
        self._load_data()
    
    def _load_data(self):
        """Charge les données depuis les fichiers"""
        # 1. Charger les métiers
        jobs_file = self.data_path / "jobs.json"
        if jobs_file.exists():
            with open(jobs_file, 'r', encoding='utf-8') as f:
                self.jobs = json.load(f).get("jobs", [])
            logger.info(f"{len(self.jobs)} métiers chargés")
        
        # 2. Charger les catégories
        categories_file = self.data_path / "job_categories.json"
        if categories_file.exists():
            with open(categories_file, 'r') as f:
                self.job_categories = json.load(f)
        
        # 3. Charger les relations
        relations_file = self.data_path / "skill_relations.json"
        if relations_file.exists():
            with open(relations_file, 'r') as f:
                self.skill_relationships = json.load(f)
    
    def get_job_by_id(self, job_id: str) -> Optional[Dict]:
        """Retrouve un métier par son ID"""
        for job in self.jobs:
            if job.get("id") == job_id:
                return job
        return None
    
    def get_jobs_by_category(self, category: str) -> List[Dict]:
        """Retrouve les métiers d'une catégorie"""
        return [job for job in self.jobs if job.get("category") == category]
    
    def get_jobs_requiring_skill(self, skill_id: str) -> List[Dict]:
        """Retrouve les métiers nécessitant une compétence"""
        return [job for job in self.jobs if skill_id in job.get("required_competencies", [])]
    
    def search_jobs(self, query: str, field: str = "title") -> List[Dict]:
        """Recherche des métiers"""
        query_lower = query.lower()
        results = []
        
        for job in self.jobs:
            if query_lower in job.get(field, "").lower():
                results.append(job)
        
        return results
    
    def get_similar_jobs(self, job_id: str, max_results: int = 5) -> List[Dict]:
        """Trouve des métiers similaires"""
        target_job = self.get_job_by_id(job_id)
        if not target_job:
            return []
        
        target_skills = set(target_job.get("required_competencies", []))
        
        similarities = []
        for job in self.jobs:
            if job["id"] == job_id:
                continue
            
            job_skills = set(job.get("required_competencies", []))
            
            # Similarité de Jaccard
            intersection = len(target_skills.intersection(job_skills))
            union = len(target_skills.union(job_skills))
            
            similarity = intersection / union if union > 0 else 0.0
            similarities.append((job, similarity))
        
        # Trier par similarité décroissante
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [job for job, _ in similarities[:max_results]]
    
    def validate_job_data(self, job: Dict) -> List[str]:
        """Valide les données d'un métier"""
        errors = []
        
        # Vérifications
        if "id" not in job:
            errors.append("ID manquant")
        if "title" not in job or not job["title"].strip():
            errors.append("Titre manquant")
        if "required_competencies" not in job:
            errors.append("Compétences requises manquantes")
        elif not isinstance(job["required_competencies"], list):
            errors.append("Compétences requises doit être une liste")
        
        return errors
    
    def add_job(self, job: Dict) -> bool:
        """Ajoute un nouveau métier"""
        errors = self.validate_job_data(job)
        if errors:
            logger.error(f"Erreurs de validation: {errors}")
            return False
        
        # Vérifier si l'ID existe déjà
        if self.get_job_by_id(job["id"]):
            logger.error(f"ID {job['id']} existe déjà")
            return False
        
        self.jobs.append(job)
        logger.info(f"Métier ajouté: {job['title']}")
        
        return True
    
    def save_to_file(self, filepath: Optional[str] = None):
        """Sauvegarde les données dans un fichier"""
        if filepath is None:
            filepath = self.data_path / "jobs_updated.json"
        
        data = {"jobs": self.jobs}
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Data saved in {filepath}")
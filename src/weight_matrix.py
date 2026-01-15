"""
weight_matrix.py - Matrice de poids pour les scores par métier
"""

# Jobs IDs
JOBS = {
    "M1": "Data Scientist",
    "M2": "Data Analyst",
    "M3": "Data Engineer",
    "M4": "ML Engineer",
    "M5": "AI Specialist",
    "M6": "Cloud Engineer",
    "M7": "Statistical Analyst",
    "M8": "Business Data Analyst",
    "M9": "BI Developer",
    "M10": "NLP Engineer",
    "M11": "Big Data Engineer",
    "M12": "Software Engineer",
    "M13": "Statistician"
}

# Domain IDs avec mapping blocs
DOMAINS = {
    "1": "Data Analysis",
    "2": "Programming & Software Development",
    "3": "Machine Learning",
    "4": "Data Engineering",
    "5": "Cloud & DevOps",
    "6": "NLP & AI Advanced"
}

# Normalized matrix (0-1) using job IDs as rows and domain IDs as columns
MATRIX = {
    "M1": {"1": 1.0, "2": 0.8, "3": 1.0, "4": 0.6, "5": 0.4, "6": 0.6},
    "M2": {"1": 1.0, "2": 0.4, "3": 0.2, "4": 0.4, "5": 0.2, "6": 0.0},
    "M3": {"1": 0.4, "2": 0.8, "3": 0.2, "4": 1.0, "5": 0.8, "6": 0.0},
    "M4": {"1": 0.6, "2": 0.8, "3": 1.0, "4": 0.6, "5": 0.6, "6": 0.6},
    "M5": {"1": 0.6, "2": 0.8, "3": 1.0, "4": 0.4, "5": 0.4, "6": 1.0},
    "M6": {"1": 0.2, "2": 0.6, "3": 0.0, "4": 0.6, "5": 1.0, "6": 0.0},
    "M7": {"1": 1.0, "2": 0.4, "3": 0.2, "4": 0.2, "5": 0.0, "6": 0.0},
    "M8": {"1": 1.0, "2": 0.4, "3": 0.2, "4": 0.4, "5": 0.2, "6": 0.0},
    "M9": {"1": 0.8, "2": 0.6, "3": 0.2, "4": 0.8, "5": 0.4, "6": 0.0},
    "M10": {"1": 0.4, "2": 0.8, "3": 1.0, "4": 0.4, "5": 0.4, "6": 1.0},
    "M11": {"1": 0.4, "2": 0.8, "3": 0.4, "4": 1.0, "5": 0.8, "6": 0.2},
    "M12": {"1": 0.2, "2": 1.0, "3": 0.4, "4": 0.4, "5": 0.6, "6": 0.2},
    "M13": {"1": 1.0, "2": 0.4, "3": 0.2, "4": 0.2, "5": 0.0, "6": 0.0}
}

class WeightMatrix:
    """
    Gestionnaire de la matrice de poids pour les scores par métier
    """
    
    @staticmethod
    def get_job_score(bloc_scores: dict, job_id: str) -> float:
        """
        Calcule le score pondéré pour un métier spécifique
        
        Args:
            bloc_scores: Scores par bloc {nom_bloc: score}
            job_id: ID du métier (ex: "M1")
            
        Returns:
            float: Score pondéré 0-1
        """
        if job_id not in MATRIX:
            return 0.0
        
        # Récupérer les poids pour ce métier
        job_weights = MATRIX[job_id]
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        # Pour chaque domaine dans la matrice
        for domain_id, weight in job_weights.items():
            if domain_id in DOMAINS:
                domain_name = DOMAINS[domain_id]
                
                # Chercher le score correspondant dans bloc_scores
                # On fait une recherche flexible par nom de domaine
                bloc_score = WeightMatrix._find_bloc_score(bloc_scores, domain_name)
                
                # Ajouter au score pondéré
                total_weighted_score += bloc_score * weight
                total_weight += weight
        
        # Éviter la division par zéro
        if total_weight == 0:
            return 0.0
        
        return total_weighted_score / total_weight
    
    @staticmethod
    def _find_bloc_score(bloc_scores: dict, domain_name: str) -> float:
        """
        Trouve le score d'un bloc en faisant une recherche flexible
        
        Args:
            bloc_scores: Scores par bloc
            domain_name: Nom du domaine (peut être différent des noms de blocs)
            
        Returns:
            float: Score trouvé ou 0.0
        """
        # Mapping direct
        if domain_name in bloc_scores:
            return bloc_scores[domain_name]
        
        # Recherche flexible par mots clés
        domain_lower = domain_name.lower()
        
        for bloc_name, score in bloc_scores.items():
            bloc_lower = bloc_name.lower()
            
            # Vérifier les correspondances
            if ("data analysis" in domain_lower and "data analysis" in bloc_lower) or \
               ("programming" in domain_lower and "python" in bloc_lower) or \
               ("machine learning" in domain_lower and "machine learning" in bloc_lower) or \
               ("data engineering" in domain_lower and "data engineering" in bloc_lower) or \
               ("cloud" in domain_lower and "cloud" in bloc_lower) or \
               ("nlp" in domain_lower and "nlp" in bloc_lower):
                return score
        
        # Si aucune correspondance trouvée
        return 0.0
    
    @staticmethod
    def get_recommendations(bloc_scores: dict, top_k: int = 3) -> list:
        """
        Recommande des métiers basés sur les scores pondérés
        
        Args:
            bloc_scores: Scores par bloc
            top_k: Nombre de recommandations
            
        Returns:
            list: Métiers recommandés avec scores
        """
        job_scores = []
        
        # Calculer le score pour chaque métier
        for job_id, job_name in JOBS.items():
            score = WeightMatrix.get_job_score(bloc_scores, job_id)
            job_scores.append({
                "id": job_id,
                "name": job_name,
                "score": score,
                "weights": MATRIX.get(job_id, {})
            })
        
        # Trier par score décroissant
        job_scores.sort(key=lambda x: x["score"], reverse=True)
        
        return job_scores[:top_k]
    
    @staticmethod
    def explain_score(job_id: str, bloc_scores: dict) -> str:
        """
        Génère une explication détaillée du score
        
        Args:
            job_id: ID du métier
            bloc_scores: Scores par bloc
            
        Returns:
            str: Explication textuelle
        """
        if job_id not in MATRIX:
            return "Métier non trouvé"
        
        job_weights = MATRIX[job_id]
        explanations = []
        
        for domain_id, weight in job_weights.items():
            if domain_id in DOMAINS:
                domain_name = DOMAINS[domain_id]
                bloc_score = WeightMatrix._find_bloc_score(bloc_scores, domain_name)
                
                # Importance du domaine pour ce métier
                importance = "Haute" if weight >= 0.8 else "Moyenne" if weight >= 0.4 else "Basse"
                
                explanations.append(
                    f"- {domain_name}: Votre score {bloc_score:.1%} × Poids {importance} ({weight})"
                )
        
        return "\n".join(explanations)
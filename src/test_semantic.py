# test_semantic.py
import sys
sys.path.append('src')

from semantic_engine import SemanticEngine

# Données de test
user_responses = {
    "q1_description": "Je développe des modèles de machine learning en Python avec scikit-learn et TensorFlow.",
    "q2_projects": "J'ai travaillé sur un système de recommandation utilisant des embeddings et de la similarité cosinus.",
    "technologies": ["Python", "TensorFlow", "scikit-learn", "Pandas"]
}

competency_blocks = {
    "Data Analysis": [
        "analyse exploratoire des données",
        "visualisation de données",
        "nettoyage de données",
        "programmation Python pour l'analyse"
    ],
    "Machine Learning": [
        "modèles de classification",
        "modèles de régression",
        "réseaux de neurones",
        "évaluation de modèles"
    ],
    "NLP": [
        "traitement du langage naturel",
        "embeddings de mots",
        "modèles transformers",
        "analyse sémantique"
    ]
}

# Test
engine = SemanticEngine()
scores, overall = engine.analyze_responses(user_responses, competency_blocks)

print("=== RÉSULTATS ===")
for block, score in scores.items():
    print(f"{block}: {score:.3f} ({score*100:.1f}%)")
print(f"\nScore global: {overall:.3f} ({overall*100:.1f}%)")
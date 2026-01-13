# test_genai.py
import os
from dotenv import load_dotenv
from genai_handler import GenAIHandler

load_dotenv()

# Test
handler = GenAIHandler()

# Contexte test
context = {
    "user_profile": {"name": "Test", "experience_years": 2},
    "block_scores": {"Data Analysis": 0.8, "ML": 0.3},
    "recommended_jobs": [{"title": "Data Scientist"}]
}

# Test plan (premier appel = API, deuxième = cache)
print("=== Test Plan de Progression ===")
plan1 = handler.generate_progression_plan(context)
print(f"Longueur: {len(plan1)} caractères")

plan2 = handler.generate_progression_plan(context)  # Doit venir du cache
print(f"Depuis cache: {plan1 == plan2}")

# Stats
print(f"\nStats: {handler.get_stats()}")
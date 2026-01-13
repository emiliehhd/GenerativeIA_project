"""
PromptTemplates - Templates optimisés pour Gemini
"""

PROGRESSION_PLAN_TEMPLATE = """
En tant que conseiller carrière IA, crée un plan sur mesure.

CONTEXTE:
{context}

EXIGENCES:
1. Sois PRÉCIS (noms de cours, durées, plateformes)
2. Sois RÉALISTE (timeline réalisable)
3. Sois ACTIONNABLE (étapes concrètes)
4. Inclus des METRICS (comment mesurer le progrès)
5. MAX 250 mots

FORMAT:
🎯 **Objectif à 6 mois**
📚 **Formations prioritaires** (avec liens si possible)
🛠️ **Projets pratiques** (2-3 idées)
📈 **Indicateurs de progression**
💡 **Conseils complémentaires**
"""

BIO_PROFESSIONAL_TEMPLATE = """
En tant que expert RH tech, rédige une bio percutante.

PROFIL:
{profile}

COMPÉTENCES:
{skills}

CONTRAINTES:
- Style "Executive Summary"
- 80 mots MAXIMUM
- Phrases courtes et impactantes
- Inclure: valeur ajoutée + expertise + aspiration
- Format LinkedIn optimisé

TONES possibles:
- "Data-driven" pour les analystes
- "Innovant" pour les ingénieurs ML
- "Stratégique" pour les managers
"""
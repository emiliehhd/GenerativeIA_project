# AISCA - Agent Intelligent Sémantique pour la Cartographie des Compétences

## Description

AISCA est une application web intelligente d'analyse sémantique des compétences et de recommandation de métiers, développée dans le cadre du projet IA Générative.

## Fonctionnalités

- Analyse sémantique avancée avec SBERT
- Matrice de poids personnalisée par métier
- Recommandations personnalisées de métiers
- Génération de plans de progression avec GenAI
- Interface web interactive avec Streamlit
- Cache pour optimisation des performances

## Architecture

src/ → Moteurs d'analyse<br>
app/ → Interface Streamlit<br>
data/ → Référentiels structurés<br>
docs/ → Documentation<br>

## Installation 

Cloner le repository
```bash
git clone https://github.com/emiliehhd/GenerativeIA_project.git
```

Créer un environnement virtuel et l'activer
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```
ou sous windows
```bash
venv\Scripts\activate     # Windows
```

Installer les dépendances
```bash
pip install -r requirements.txt
```

Configurer l'environnement
```bash
cp .env.example .env
# Éditer .env avec votre cle API
```

Lancer l'application
```bash
streamlit run app/main.py

```

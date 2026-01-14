"""
Test du module semantic_engine.py
Ce fichier permet de tester le moteur d'analyse sémantique indépendamment de l'interface Streamlit
"""

import sys
import os
import pandas as pd
import numpy as np

# Ajouter le dossier parent au path pour pouvoir importer les modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from semantic_engine import SemanticEngine
    print("✅ Module semantic_engine importé avec succès")
except ImportError as e:
    print(f"❌ Erreur d'importation: {e}")
    print("Assurez-vous que le fichier semantic_engine.py existe dans le dossier src/")
    sys.exit(1)

def create_test_data():
    """Crée des données de test si les fichiers n'existent pas"""
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Créer referentiel_competences.csv si nécessaire
    competences_path = os.path.join(data_dir, "referentiel_competences.csv")
    if not os.path.exists(competences_path):
        competences_data = {
            'competence_id': ['C001', 'C002', 'C003', 'C004', 'C005', 'C006', 'C007', 'C008', 'C009', 'C010'],
            'competence': [
                'Python programmation',
                'SQL querying', 
                'Data visualization',
                'Statistical analysis',
                'Machine Learning algorithms',
                'Deep Learning',
                'Model evaluation',
                'NLP basics',
                'Text preprocessing',
                'Word embeddings'
            ],
            'bloc': [
                'Data Analysis',
                'Data Analysis', 
                'Data Analysis',
                'Data Analysis',
                'Machine Learning',
                'Machine Learning',
                'Machine Learning',
                'NLP',
                'NLP',
                'NLP'
            ],
            'description': [
                'Programmation en Python pour l analyse de donnees',
                'Requetes SQL pour l extraction de donnees',
                'Creation de visualisations avec Matplotlib/Seaborn',
                'Analyse statistique des donnees',
                'Algorithmes de machine learning supervise',
                'Reseaux de neurones et deep learning',
                'Evaluation et validation des modeles',
                'Base du traitement du langage naturel',
                'Pretraitement de texte',
                'Embeddings de mots et representations vectorielles'
            ]
        }
        df_competences = pd.DataFrame(competences_data)
        df_competences.to_csv(competences_path, index=False, encoding='utf-8')
        print(f"✅ Fichier créé: {competences_path}")
    
    # Créer referentiel_metier.csv si nécessaire
    metiers_path = os.path.join(data_dir, "referentiel_metier.csv")
    if not os.path.exists(metiers_path):
        metiers_data = {
            'metier_id': ['M001', 'M002', 'M003', 'M004', 'M005'],
            'metier': [
                'Data Scientist',
                'Data Analyst', 
                'Machine Learning Engineer',
                'Data Engineer',
                'NLP Engineer'
            ],
            'competences_requises': [
                'C001;C002;C003;C004;C005;C006;C007;C008',
                'C001;C002;C003;C004',
                'C001;C005;C006;C007',
                'C001;C002',
                'C001;C008;C009;C010;C005'
            ],
            'description': [
                'Analyse de donnees complexes et developpement de modeles predictifs',
                'Analyse descriptive et creation de rapports business',
                'Ingenierie et deploiement de modeles ML',
                'Construction et maintenance de pipelines de donnees',
                'Specialiste en traitement du langage naturel'
            ],
            'experience_min': [2, 0, 2, 2, 1],
            'niveau': ['Senior', 'Junior', 'Senior', 'Senior', 'Mid']
        }
        df_metiers = pd.DataFrame(metiers_data)
        df_metiers.to_csv(metiers_path, index=False, encoding='utf-8')
        print(f"✅ Fichier créé: {metiers_path}")
    
    return competences_path, metiers_path

def test_initialization():
    """Test l'initialisation du SemanticEngine"""
    print("\n" + "="*60)
    print("TEST 1: Initialisation du SemanticEngine")
    print("="*60)
    
    try:
        engine = SemanticEngine(model_name="all-MiniLM-L6-v2")
        print("✅ SemanticEngine initialisé avec succès")
        print(f"   Modèle utilisé: all-MiniLM-L6-v2")
        return engine
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        return None

def test_data_loading(engine):
    """Test le chargement des données"""
    print("\n" + "="*60)
    print("TEST 2: Chargement des données")
    print("="*60)
    
    competences_path, metiers_path = create_test_data()
    
    try:
        engine.load_data(competences_path, metiers_path)
        print("✅ Données chargées avec succès")
        
        print(f"\n📊 Compétences chargées:")
        print(f"   - Nombre de compétences: {len(engine.competences_df)}")
        print(f"   - Blocs disponibles: {engine.competences_df['bloc'].unique().tolist()}")
        
        print(f"\n📊 Métiers chargés:")
        print(f"   - Nombre de métiers: {len(engine.metiers_df)}")
        print(f"   - Métiers disponibles: {engine.metiers_df['metier'].tolist()}")
        
        return True
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données: {e}")
        return False

def test_analysis(engine):
    """Test l'analyse sémantique avec différents profils"""
    print("\n" + "="*60)
    print("TEST 3: Analyse sémantique de profils utilisateur")
    print("="*60)
    
    # Profil 1: Débutant en data
    print("\n🧪 PROFIL 1: Débutant en data")
    user_responses_1 = {
        'competences': "Je connais un peu Python pour analyser des données et faire des graphiques simples. J'ai appris les bases de SQL.",
        'experiences': "J'ai fait un projet universitaire où j'ai analysé des données avec Python. J'ai utilisé Pandas et Matplotlib.",
        'outils': "Python, Pandas, Matplotlib, SQL"
    }
    
    try:
        results_1 = engine.analyze_user_profile(user_responses_1)
        print("✅ Analyse réussie pour le profil débutant")
        display_results(results_1, "Débutant")
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")
    
    # Profil 2: Intermédiaire en machine learning
    print("\n🧪 PROFIL 2: Intermédiaire en machine learning")
    user_responses_2 = {
        'competences': "Je maîtrise Python pour le machine learning. Je connais Scikit-learn, les algorithmes de classification et de régression.",
        'experiences': "J'ai développé plusieurs modèles de prédiction en entreprise. J'ai travaillé sur des projets de classification d'images.",
        'outils': "Python, Scikit-learn, TensorFlow, Pandas, NumPy, Jupyter"
    }
    
    try:
        results_2 = engine.analyze_user_profile(user_responses_2)
        print("✅ Analyse réussie pour le profil intermédiaire")
        display_results(results_2, "Intermédiaire")
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")
    
    # Profil 3: Expert en NLP
    print("\n🧪 PROFIL 3: Expert en NLP")
    user_responses_3 = {
        'competences': "Expert en traitement du langage naturel, embeddings de mots, modèles transformers, analyse sémantique.",
        'experiences': "J'ai développé des chatbots intelligents, des systèmes de classification de texte, et travaillé avec BERT et GPT.",
        'outils': "Python, TensorFlow, PyTorch, Transformers, SpaCy, NLTK, BERT, GPT"
    }
    
    try:
        results_3 = engine.analyze_user_profile(user_responses_3)
        print("✅ Analyse réussie pour le profil expert")
        display_results(results_3, "Expert")
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")

def display_results(results, profile_name):
    """Affiche les résultats de l'analyse de manière lisible"""
    print(f"\n📈 RÉSULTATS pour {profile_name}:")
    
    # Score global
    overall_score = results.get('overall_score', 0) * 100
    print(f"   Score global: {overall_score:.1f}%")
    
    # Scores par bloc
    if 'bloc_scores' in results and results['bloc_scores']:
        print(f"   Scores par domaine:")
        for bloc, score in results['bloc_scores'].items():
            print(f"   - {bloc}: {score*100:.1f}%")
    
    # Métiers recommandés
    if 'metier_recommendations' in results and results['metier_recommendations']:
        print(f"   Métiers recommandés (top 3):")
        for i, metier in enumerate(results['metier_recommendations'][:3], 1):
            print(f"   {i}. {metier['metier']} ({metier['score']*100:.1f}%)")
    
    # Compétences fortes
    if 'strong_competences' in results and results['strong_competences']:
        strong_count = len(results['strong_competences'])
        print(f"   Compétences fortes identifiées: {strong_count}")
        if strong_count > 0:
            print(f"   Exemples: {results['strong_competences'][0]['competence']}")
    
    # Compétences faibles
    if 'weak_competences' in results and results['weak_competences']:
        weak_count = len(results['weak_competences'])
        print(f"   Compétences à améliorer: {weak_count}")

def test_edge_cases(engine):
    """Test des cas limites"""
    print("\n" + "="*60)
    print("TEST 4: Cas limites")
    print("="*60)
    
    # Test 1: Réponses vides
    print("\n🧪 CAS 1: Réponses vides")
    empty_responses = {
        'competences': "",
        'experiences': "",
        'outils': ""
    }
    
    try:
        results = engine.analyze_user_profile(empty_responses)
        print("✅ Gestion des réponses vides: OK")
        if results.get('overall_score', 0) < 0.1:
            print("   Score bas comme attendu pour réponses vides")
    except Exception as e:
        print(f"❌ Erreur avec réponses vides: {e}")
    
    # Test 2: Réponses très courtes
    print("\n🧪 CAS 2: Réponses très courtes")
    short_responses = {
        'competences': "Python",
        'experiences': "Stage",
        'outils': "Excel"
    }
    
    try:
        results = engine.analyze_user_profile(short_responses)
        print("✅ Gestion des réponses courtes: OK")
    except Exception as e:
        print(f"❌ Erreur avec réponses courtes: {e}")
    
    # Test 3: Texte très long
    print("\n🧪 CAS 3: Texte très long")
    long_text = "Python " * 100
    long_responses = {
        'competences': long_text,
        'experiences': long_text,
        'outils': long_text
    }
    
    try:
        results = engine.analyze_user_profile(long_responses)
        print("✅ Gestion des textes longs: OK")
    except Exception as e:
        print(f"❌ Erreur avec texte long: {e}")

def test_performance(engine):
    """Test des performances"""
    print("\n" + "="*60)
    print("TEST 5: Tests de performance")
    print("="*60)
    
    import time
    
    # Test de performance avec un profil moyen
    test_responses = {
        'competences': "Python, machine learning, data analysis, SQL, statistics",
        'experiences': "2 ans d'expérience en analyse de données, projets en machine learning",
        'outils': "Python, Pandas, Scikit-learn, TensorFlow, SQL, Git"
    }
    
    # Mesurer le temps d'exécution
    print("\n⏱️  Mesure du temps d'exécution:")
    
    start_time = time.time()
    try:
        results = engine.analyze_user_profile(test_responses)
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"✅ Analyse terminée en {execution_time:.2f} secondes")
        
        if execution_time < 5:
            print("   ⚡ Performance: Excellente")
        elif execution_time < 10:
            print("   ✅ Performance: Correcte")
        else:
            print("   ⚠️  Performance: Lente, à optimiser")
            
    except Exception as e:
        print(f"❌ Erreur lors du test de performance: {e}")

def run_comprehensive_tests():
    """Exécute tous les tests"""
    print("🧪 LANCEMENT DES TESTS COMPLETS POUR semantic_engine.py")
    print("="*60)
    
    # Test 1: Initialisation
    engine = test_initialization()
    if engine is None:
        print("❌ Impossible de continuer les tests sans SemanticEngine")
        return
    
    # Test 2: Chargement des données
    if not test_data_loading(engine):
        print("❌ Impossible de continuer sans données chargées")
        return
    
    # Test 3: Analyse sémantique
    test_analysis(engine)
    
    # Test 4: Cas limites
    test_edge_cases(engine)
    
    # Test 5: Performance
    test_performance(engine)
    
    print("\n" + "="*60)
    print("🎯 TESTS TERMINÉS")
    print("="*60)

def simple_test():
    """Version simplifiée du test pour un usage rapide"""
    print("🧪 TEST SIMPLIFIÉ DE semantic_engine.py")
    print("="*60)
    
    try:
        # Initialiser
        engine = SemanticEngine()
        print("✅ SemanticEngine initialisé")
        
        # Charger les données
        competences_path, metiers_path = create_test_data()
        engine.load_data(competences_path, metiers_path)
        print("✅ Données chargées")
        
        # Tester avec un profil simple
        test_responses = {
            'competences': "Je connais Python pour l'analyse de données et SQL pour les bases de données",
            'experiences': "Projet universitaire en data science, stage en entreprise",
            'outils': "Python, Pandas, SQL, Git"
        }
        
        print("\n🧪 Analyse d'un profil utilisateur...")
        results = engine.analyze_user_profile(test_responses)
        
        # Afficher les résultats
        print("\n📊 RÉSULTATS:")
        print(f"Score global: {results.get('overall_score', 0)*100:.1f}%")
        
        if 'metier_recommendations' in results:
            print("\n🎯 Métiers recommandés:")
            for metier in results['metier_recommendations'][:3]:
                print(f"  - {metier['metier']} ({metier['score']*100:.1f}%)")
        
        print("\n✅ Test réussi! Le moteur sémantique fonctionne correctement.")
        
    except Exception as e:
        print(f"\n❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Demander le type de test à l'utilisateur
    print("Choisissez le type de test:")
    print("1. Test complet (recommandé)")
    print("2. Test simplifié (rapide)")
    print("3. Quitter")
    
    choice = input("\nVotre choix (1-3): ").strip()
    
    if choice == "1":
        run_comprehensive_tests()
    elif choice == "2":
        simple_test()
    elif choice == "3":
        print("Au revoir!")
    else:
        print("Choix invalide. Exécution du test simplifié par défaut.")
        simple_test()
    
    print("\n" + "="*60)
    print("💡 Conseil: Lancez 'streamlit run app.py' pour tester l'interface complète")
    print("="*60)
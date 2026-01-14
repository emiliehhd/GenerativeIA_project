"""
Interface principale Streamlit 
Application 
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

from dotenv import load_dotenv
load_dotenv()  

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Importer les modules
from semantic_engine import SemanticEngine
from genai_handler import GenAIHandler

# Configuration de la page
st.set_page_config(
    page_title="Competencies mapping",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialiser les états de session
if 'user_responses' not in st.session_state:
    st.session_state.user_responses = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = "questionnaire"
if 'user_info' not in st.session_state:
    st.session_state.user_info = {
        "name": "",
        "experience": 0
    }

# Initialiser les moteurs (avec cache)
@st.cache_resource
def init_semantic_engine():
    """Initialise le moteur sémantique"""
    engine = SemanticEngine()
    engine.load_data(
        "data/referentiel_competences.csv",
        "data/referentiel_metier.csv"
    )
    return engine

@st.cache_resource
def init_genai_handler():
    """Initialise le handler GenAI"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.warning("Cle gemni introuvable. Bio par defaut")
    return GenAIHandler(api_key=api_key) 

# Fonctions d'affichage
def display_header():
    """Affiche l'en-tête de l'application"""
    st.title("Competencies mapping")
    st.markdown("Semantic analysis of your skills and career recommendations")
    st.divider()

def display_sidebar():
    """Affiche la barre latérale avec navigation"""
    with st.sidebar:
        st.markdown("### Navigation")
        
        # Boutons de navigation
        if st.button("Questionnaire", use_container_width=True, type="primary" if st.session_state.current_step == "questionnaire" else "secondary"):
            st.session_state.current_step = "questionnaire"
            st.rerun()
            
        if st.button("Results", use_container_width=True, type="primary" if st.session_state.current_step == "results" else "secondary"):
            st.session_state.current_step = "results"
            st.rerun()
            
        if st.button("Progress Plan", use_container_width=True, type="primary" if st.session_state.current_step == "progression" else "secondary"):
            st.session_state.current_step = "progression"
            st.rerun()
            
        if st.button("Professionnel Bio", use_container_width=True, type="primary" if st.session_state.current_step == "bio" else "secondary"):
            st.session_state.current_step = "bio"
            st.rerun()

        st.divider()
        st.markdown("### About us")
        st.caption("May BILAL and Emilie DOAN")
        st.caption("")
        st.caption("Generative AI Project - Semantic analysis for skills mapping")

def display_questionnaire():
    """Affiche le questionnaire principal"""
    st.header("Skills Assessment Questionnaire")
    
    with st.form("questionnaire_form"):
        # Informations utilisateur
        st.subheader("Your Informations")
        st.session_state.user_info["name"] = st.text_input("Name", value=st.session_state.user_info["name"])

        # Question 1 : Competences
        st.subheader("1. Competenciees")
        st.markdown("Describe your technical skills: programming languages, methods, areas of expertise...")
        
        competences = st.text_area(
            "Examples: Python, machine learning, statistics, SQL, data analysis, deep learning...",
            height=120,
            key="q_competences",
            placeholder="I am proficient in Python for data analysis and I have a basic understanding of machine learning..."
        )
        
        # Question 2 : Experiences et Projets
        st.subheader("2. Experiences and Projects")
        st.markdown("Describe your academic experiences, personal projects, or professional experiences.")
        
        experiences = st.text_area(
            "Exemples : Final year project in data science, internship in data analysis, personal visualization project...",
            height=120,
            key="q_experiences",
            placeholder="I completed an image classification project during my master's degree. I also did an internship where I developed analytical dashboards..."
        )

        # st.subheader("3. Auto-evaluation per Competencies")
        
        # # Competencies auto-evaluation : python
        # python_level = st.radio(
        #     "Indicate your Python level :",
        #     options=["1 - Debutant", "2 - Basics", "3 - Intermediate", "4 - Advance", "5 - Expert"],
        #     horizontal=True,
        #     key="python_likert"
        # )
        # python_score = int(python_level.split(" - ")[0]) #1 à 5

        # # Competencies auto-evaluation : SQL
        # sql_level = st.radio(
        #     "Indicate your SQL level :",
        #     options=["1 - Debutant", "2 - Basics", "3 - Intermediate", "4 - Advance", "5 - Expert"],
        #     horizontal=True,
        #     key="sql_likert"
        # )
        # sql_score = int(sql_level.split(" - ")[0])

        
        # Question 3 : Outils et Technologies
        st.subheader("3. Tools and Technologies Mastered")
        st.markdown("List the tools, software and technologies that you are familiar with")
        
        outils = st.text_area(
            "Exemples : Pandas, NumPy, TensorFlow, Tableau, Git, Docker, AWS, Spark...",
            height=120,
            key="q_outils",
            placeholder="I regularly use Pandas for data manipulation, Matplotlib for visualization, and Git for version control..."
        )
        
        # Bouton de soumission
        submitted = st.form_submit_button("Analyze", use_container_width=True)
        
        if submitted:
            # Valider les entrees
            if not competences.strip():
                st.error("Please describe your technical skills.")
                return
            
            # Stocker les reponses
            st.session_state.user_responses = {
                "competences": competences,
                # "python_score": python_score, #ICI
                # "sql_score": sql_score,
                "experiences": experiences,
                "outils": outils,
                # "python_likert_label": python_level, #ICI
                # "sql_likert_label": sql_level  
            }
            
            # Lancer l'analyse
            with st.spinner("Semantic analysis in progress..."):
                try:
                    # Initialiser les moteurs
                    engine = init_semantic_engine()
                    
                    # Analyser le profil
                    analysis_results = engine.analyze_user_profile(st.session_state.user_responses)
                    
                    # Stocker les resultats
                    st.session_state.analysis_results = analysis_results
                    
                    # Passer aux resultats
                    st.session_state.current_step = "results"
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error during analysis : {str(e)}")
                    st.info("Make sure the data files are present in the data/ folder")

def display_results():
    """Affiche les resultats de l'analyse"""
    if not st.session_state.analysis_results:
        st.warning("Please complete the questionnaire first..")
        if st.button("Back to the questionnaire"):
            st.session_state.current_step = "questionnaire"
            st.rerun()
        return
    
    results = st.session_state.analysis_results
    
    st.header("Results of the Semantic Analysis")
    
    # Score global
    col1, col2, col3 = st.columns(3)
    
    with col1:
        score_percent = results.get('overall_score', 0) * 100
        st.metric(
            label="Score Global",
            value=f"{score_percent:.1f}%"
        )
    
    with col2:
        nb_competences = len(results.get('competence_details', []))
        st.metric(label="Competencies assessed", value=nb_competences)

    
    # Graphique des scores par bloc
    if 'bloc_scores' in results and results['bloc_scores']:
        st.subheader("Scores by domain of ​​expertise")
        
        # Graphique radar
        categories = list(results['bloc_scores'].keys())
        values = [v * 100 for v in results['bloc_scores'].values()]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Scores'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tableau detaille
        st.subheader("Score Details by Domain")
        
        scores_data = []
        for bloc, score in results['bloc_scores'].items():
            scores_data.append({
                "Domain": bloc,
                "Score": f"{score * 100:.1f}%",
                "Evaluation": "High" if score >= 0.7 else "Medium" if score >= 0.5 else "Low"
            })
        
        scores_df = pd.DataFrame(scores_data)
        st.dataframe(scores_df, use_container_width=True, hide_index=True)
    
    # Recommendations de metiers
    st.divider()
    st.subheader("Recommended Jobs")
    
    recommendations = results.get('metier_recommendations', [])
    
    if recommendations:
        for i, job in enumerate(recommendations, 1):
            with st.expander(f"{i}. {job['metier']} - Score: {job['score']:.1%}", expanded=(i==1)):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Description:** {job['description']}")
                
                with col2:
                    # Barre de progression
                    progress_value = job['score']
                    st.progress(progress_value, text=f"Adéquation: {progress_value:.1%}")
                    
                    # Bouton pour generer un plan
                    if st.button(f"Plan for this job", key=f"plan_{i}"):
                        st.session_state.selected_job = job['metier']
                        st.session_state.current_step = "progression"
                        st.rerun()
    else:
        st.info("No rocommendation available.")
    
    # Boutons d'action
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("New analysis", use_container_width=True):
            st.session_state.user_responses = {}
            st.session_state.analysis_results = None
            st.session_state.current_step = "questionnaire"
            st.rerun()
    
    with col2:
        if st.button("Generate progress plan", use_container_width=True):
            st.session_state.current_step = "progression"
            st.rerun()
    
    with col3:
        if st.button("Generate bio", use_container_width=True):
            st.session_state.current_step = "bio"
            st.rerun()

def display_progression_plan():
    """Affiche le plan de progression"""
    if not st.session_state.analysis_results:
        st.error("Please complete the questionnaire first..")
        if st.button("Back to the questionnaire"):
            st.session_state.current_step = "questionnaire"
            st.rerun()
        return
    
    results = st.session_state.analysis_results
    
    st.header("Personalized Progress Plan")
    
    # Identifier les competences faibles
    weak_competences = results.get('weak_competences', [])
    
    if weak_competences:
        st.warning(f"{len(weak_competences)} competences skills identified as needing improvement")
        
        # Generer le plan avec GenAI
        with st.spinner("Generation of the progress plan..."):
            try:
                genai = init_genai_handler()
                
                # Construire le contexte
                context = {
                    **results,
                    "user_info": st.session_state.user_info
                }
                
                # Generer le plan
                progression_plan = genai.generate_progression_plan(context)
                
                # Afficher le plan
                st.markdown("### Your Development Plan")
                st.markdown(progression_plan)
                
                # Options
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Regenerate the plan", use_container_width=True):
                        st.rerun()
                with col2:
                    if st.button("???", use_container_width=True):
                        st.success("a faire ..fonction du bouton, mail?")
                
            except Exception as e:
                st.error(f"Error during generation : {e}")
                
                # Afficher un plan par defaut
                st.markdown("### Recommended Development Plan")
                
                for comp in weak_competences[:5]:
                    st.markdown(f"""
                    **{comp['competence']}** ({comp['bloc']})
                    - Suivre un cours specialise
                    - Pratiquer avec des exercices concrets
                    - Realiser un mini-projet
                    - Delai estime : 2-3 mois
                    """)
    else:
        st.success("All your skills are well mastered!")
        st.info("You might consider specializing further or exploring new advanced technologies.")
    
    # Bouton de retour
    st.divider()
    if st.button("Back to results", use_container_width=True):
        st.session_state.current_step = "results"
        st.rerun()

def display_bio():
    """Displays professional bio"""
    if not st.session_state.analysis_results:
        st.error("Please complete the questionnaire first..")
        if st.button("Back to the questionnaire"):
            st.session_state.current_step = "questionnaire"
            st.rerun()
        return
    
    results = st.session_state.analysis_results
    
    st.header("Resume Generation")
    
    # Generer la bio avec GenAI
    with st.spinner("Generating your professional bio..."):
        try:
            genai = init_genai_handler()
            
            # Generer la bio
            professional_bio = genai.generate_professional_bio(
                st.session_state.user_info,
                results
            )
            
            # Afficher la bio
            st.markdown("### Professionnel Bio")
            st.markdown(professional_bio)
            
            # Editeur pour personnalisation
            st.subheader("Personnalize")
            edited_bio = st.text_area("Modify ifneeded :", professional_bio, height=200)
            
            # Options d'export

            if st.button("Download Resume", use_container_width=True):
                st.success("lalala")
            
        except Exception as e:
            st.error(f"Error during generation : {e}")
            
            # Bio par defaut
            strong_competences = results.get('strong_competences', [])
            competences_text = ", ".join([comp['competence'] for comp in strong_competences[:3]])
            
            bio = f"""
            {st.session_state.user_info.get('name', 'Professionnel')}.
            
            Expertise en {competences_text if competences_text else 'analyse de donnees et technologies associees'}.
            
            Capacite demontree a analyser des donnees complexes et a fournir des insights actionnables.
            Experience dans la mise en œuvre de solutions basees sur les donnees.
            
            Recherche actuellement des opportunites pour contribuer a des projets innovants dans le domaine de la data.
            """
            
            st.markdown(bio)
    
    # Bouton de retour
    st.divider()
    if st.button("Back to results", use_container_width=True):
        st.session_state.current_step = "results"
        st.rerun()

# Application principale
def main():
    """Fonction principale de l'application"""
    
    # Afficher l'en-tete
    display_header()
    
    # Sidebar
    display_sidebar()
    
    # Router vers la page appropriee
    if st.session_state.current_step == "questionnaire":
        display_questionnaire()
    elif st.session_state.current_step == "results":
        display_results()
    elif st.session_state.current_step == "progression":
        display_progression_plan()
    elif st.session_state.current_step == "bio":
        display_bio()
    
    # Footer
    st.divider()
    st.caption("Generative AI Project")

if __name__ == "__main__":
    main()
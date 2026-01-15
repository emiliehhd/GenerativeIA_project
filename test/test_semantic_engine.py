"""
Tests unitaires pour SemanticEngine
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from src.semantic_engine import SemanticEngine


class TestSemanticEngine:
    """Tests pour le moteur sémantique"""
    
    @pytest.fixture
    def mock_engine(self):
        """Crée un moteur mocké pour les tests"""
        with patch('sentence_transformers.SentenceTransformer') as mock_model:
            engine = SemanticEngine()
            engine.model = mock_model
            engine.competences_df = pd.DataFrame({
                'competence': ['Python programming', 'SQL querying'],
                'bloc': ['Data Analysis', 'Data Analysis'],
                'description': ['Desc1', 'Desc2']
            })
            return engine
    
    def test_load_data(self, mock_engine, tmp_path):
        """Test le chargement des données"""
        # Créer des fichiers CSV temporaires
        competences_file = tmp_path / "competences.csv"
        metiers_file = tmp_path / "metiers.csv"
        
        competences_file.write_text("competence,bloc,description\nPython,DA,Desc")
        metiers_file.write_text("metier_id,metier,description\nM1,Data Scientist,Desc")
        
        # Tester
        mock_engine.load_data(str(competences_file), str(metiers_file))
        
        assert mock_engine.competences_df is not None
        assert mock_engine.metiers_df is not None
    
    def test_calculate_bloc_scores(self, mock_engine):
        """Test le calcul des scores par bloc"""
        mock_engine.competences_df['similarity_score'] = [0.8, 0.6]
        
        scores = mock_engine._calculate_bloc_scores()
        
        assert 'Data Analysis' in scores
        assert 0.6 <= scores['Data Analysis'] <= 0.8
    

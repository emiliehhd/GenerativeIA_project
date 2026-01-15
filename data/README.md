# Données 

## Structure

### `referentiel_competences.csv`
- **competence_id** : Identifiant unique (C1, C2...)
- **competence** : Nom de la compétence
- **bloc** : Bloc de compétences (Data Analysis, ML...)
- **description** : Description de la compétence

### `referentiel_metier.csv`
- **metier_id** : Identifiant unique (M1, M2...)
- **metier** : Nom du métier
- **competences_requises** : Liste d'IDs séparés par ;
- **description** : Description du métier

## Mise à jour

1. Ajouter de nouvelles compétences dans `referentiel_competences.csv`
2. Mettre à jour les métiers dans `referentiel_metier.csv`
3. Mettre à jour la matrice de poids dans `src/weight_matrix.py`

## Sources

- ROME : Répertoire Opérationnel des Métiers et des Emplois
- ESCO : European e-Competence Framework
- Référentiels métiers du numérique

# Scripts

Ce dossier contient divers scripts d'analyse et de génération de graphiques.

## Scripts principaux

### 📊 Analyse des sondages

- **`main.py`** - Script d'analyse générale des sondages MJ
- **`main_approval.py`** - Analyse spécifique pour le vote par approbation
- **`main_ipsos.py`** - Traitement des données IPSOS

### 🎬 Visualisation

- **`main_animation.py`** - Génération d'animations des profils de mérite (probablement obsolete)
- **`sankey_plot.py`** - Diagrammes de Sankey (en cours de construtction)
- **`sankey_v2.py`** - Diagrammes de Sankey (en cours de construtction)

### 🔧 Utilitaires

- **`main_filtering.py`** - Analyse des sondages filtrés temporellement

## Script de production

Le script principal pour la génération automatique des graphiques est **`main_export.py`** (à la racine du projet).

Il est utilisé par le workflow GitHub Actions pour publier automatiquement les données et graphiques.

## Usage

La plupart de ces scripts utilisent l'environnement conda défini dans `environment.yml` :

```bash
# Activer l'environnement
conda env create -f environment.yml
conda activate mieuxvoter

# Lancer un script
python scripts/main_ipsos.py
```

## Note

Certains scripts peuvent être obsolètes ou en cours de refactoring. Consultez le code source pour plus de détails.

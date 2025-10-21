# mj-tracker-2027

Suivi de l'opinion publique via le **Jugement Majoritaire** pour l'élection présidentielle française de 2027.

> 📚 **Projet précédent** : [mj-tracker-2022](https://github.com/MieuxVoter/majority-judgment-tracker) - Tracker développé pour la présidentielle 2022, qui a permis pour la première fois d'agréger et visualiser des sondages au jugement majoritaire durant une campagne présidentielle. Ce projet a démontré l'intérêt de suivre l'évolution de l'opinion avec une méthode de scrutin plus riche que le scrutin uninominal.

## 📊 Présentation

Ce projet analyse les sondages d'opinion en appliquant la méthode du **Jugement Majoritaire** (MJ) et du **Vote par Approbation**, permettant une évaluation plus nuancée des candidats que les sondages traditionnels d'intention de vote.

Le Jugement Majoritaire permet de :
- 📈 **Mesurer l'adhésion** réelle aux candidats, pas seulement l'intention de vote stratégique
- 🎯 **Détecter les progressions** ou dévaluations des candidats au-delà des transferts de voix
- 📊 **Révéler des candidats sous-estimés** par le scrutin uninominal (comme Fabien Roussel en 2022)
- ⚖️ **Identifier le rejet** d'un candidat malgré des intentions de vote élevées (comme Éric Zemmour en 2022)

### Sources de données

Ce projet s'appuie sur plusieurs bases de données complémentaires :

**🎯 Base principale** : [mj-database-2027](https://github.com/MieuxVoter/mj-database-2027)
- Sondages spécifiques au Jugement Majoritaire (IPSOS, ELABE, IFOP)
- Données standardisées avec échelles de satisfaction/appréciation
- Segmentation par électorat (ensemble, gauche, macronistes, extrême-droite, abstentionnistes)

**📊 Données complémentaires** : [presidentielle2027](https://github.com/MieuxVoter/presidentielle2027)
- Sondages d'intention de vote classiques (scrutin uninominal)
- Permet la comparaison entre JM et intentions de vote traditionnelles
- Compilation multi-instituts standardisée

## 🚀 Installation

### Setup

```bash
# Cloner le repo
git clone https://github.com/MieuxVoter/mj-tracker-2027.git
cd mj-tracker-2027

# Créer l'environnement conda
conda env create -f environment.yml
conda activate mieuxvoter

# Installer le package
pip install -e .
```

## 📈 Usage

### Génération automatique des graphiques

Le script principal `main_export.py` génère tous les graphiques et exports de données :

```bash
python main_export.py --png --json --csv <url_ou_fichier_csv>
```

**Options** :
- `--png` : Exporte les graphiques en PNG
- `--json` : Exporte les données en JSON
- `--svg` : Exporte les graphiques en SVG
- `--csv <path>` : Source des données (URL ou fichier local)
- `--dest <path>` : Dossier de destination (défaut: `output/`)

**Résultat** :
- Dossier `mj/` : Graphiques et données pour le Jugement Majoritaire
- Dossier `approval/` : Graphiques et données pour le Vote par Approbation

### Autres scripts

Voir [`scripts/README.md`](scripts/README.md) pour les scripts d'analyse spécifiques.

## 📦 Structure du projet

```
mj-tracker-2027/
├── main_export.py           # Script principal de génération
├── mjtracker/               # Package Python principal
│   ├── survey_interface.py  # Interface pour un sondage
│   ├── surveys_interface.py # Interface pour plusieurs sondages
│   ├── export_compact_json.py # Export JSON optimisé
│   ├── batch_plots.py       # Génération batch de graphiques
│   ├── plots_v2.py          # Fonctions de plotting
│   └── libs/                # Algorithmes de Jugement Majoritaire
├── scripts/                 # Scripts d'analyse divers
├── tests/                   # Tests unitaires
└── .github/workflows/       # CI/CD (publication automatique)
```

## 🔄 Workflow automatique

Chaque dimanche à 3h UTC, un workflow GitHub Actions :
1. Télécharge les dernières données depuis `mj-database-2027`
2. Génère tous les graphiques (MJ + Approbation)
3. Publie une release avec :
   - Archive ZIP complète (`data-export.zip`)
   - JSON compacts optimisés (`latest_survey_*_compact.json`)
   - CSV exportés (`latest_survey_*.csv`)

**Accès aux données** : [Releases](https://github.com/MieuxVoter/mj-tracker-2027/releases/tag/latest-data)

## 📊 Formats de sortie

### JSON Compact
Structure optimisée (~70% plus petit) avec :
- Métadonnées centralisées
- Types de sondage normalisés
- Distributions en tableaux

### CSV
Format tabulaire complet pour analyse externe.

### Graphiques
- Profils de mérite temporels
- Rankings des candidats
- Comparaisons par institut de sondage

## 🤝 Contribution

Les contributions sont les bienvenues ! Voir les issues pour les améliorations en cours.

## 📄 Licence

Voir LICENSE

## 🔗 Liens utiles

### Projets connexes
- 📊 [mj-database-2027](https://github.com/MieuxVoter/mj-database-2027) - Base de données des sondages compatibles JM
- 📈 [presidentielle2027](https://github.com/MieuxVoter/presidentielle2027) - Sondages d'intention de vote classiques
- 🗳️ [mj-tracker-2022](https://github.com/MieuxVoter/majority-judgment-tracker) - Projet précédent (présidentielle 2022)

### 📚 Enseignements de 2022

Le projet [mj-tracker-2022](https://github.com/MieuxVoter/majority-judgment-tracker) a révélé des insights majeurs :

#### Candidats sous-estimés
**Fabien Roussel** : Classé 4e-5e au JM avec seulement 3-5% d'intentions de vote
- Le JM a capturé son adhésion croissante invisible au scrutin uninominal
- Progression constante de sa base d'adhésion durant la campagne

#### Progressions mesurées
**Marine Le Pen** et **Jean-Luc Mélenchon** : Progressions fortes confirmées par JM et scrutin uninominal
- MLP : 7e → 3e → 2e au JM (même classement final qu'au scrutin uninominal)
- JLM : 10e → 8e → 4e au JM (vs 3e au scrutin uninominal)

#### Dévaluations détectées
**Valérie Pécresse** : Seule candidate fortement dévaluée
- Passage de 1ère à 3e position au JM
- Le JM a permis de comprendre que sa baisse n'était pas un report de voix mais une dévaluation réelle

#### Sur-valorisation révélée
**Éric Zemmour** : 4e au scrutin uninominal, dernier (12e) au JM
- Toujours rejeté par >50% des électeurs
- Démonstration que le scrutin uninominal peut sur-valoriser un candidat massivement rejeté

#### Agrégation de sondages
Premier tracker à agréger différents instituts (Opinion Way, ELABE, IFOP) :
- Standardisation des mentions entre instituts
- Moyenne glissante sur 14 jours pour lisser les données
- Visualisation de l'évolution temporelle des profils de mérite

### Ressources Mieux Voter
- 🌐 [Mieux Voter](https://mieuxvoter.fr/) - Association promouvant le Jugement Majoritaire
- 📖 [Wiki Jugement Majoritaire](https://fr.wikipedia.org/wiki/Jugement_majoritaire) - Wikipédia
- 🎓 [Théorie du JM](https://mitpress.mit.edu/9780262015134/majority-judgment/) - Livre de référence (Balinski & Laraki)

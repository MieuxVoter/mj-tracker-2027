# mjtracker Package Structure

Package Python pour le suivi et l'analyse des intentions de vote en Jugement Majoritaire.

## Utilisation

### Imports de base

```python
# Interfaces principales
from mjtracker import SurveyInterface, SurveysInterface, SMPData

# Enums et constantes
from mjtracker import AggregationMode, PollingOrganizations, Candidacy

# Utilitaires
from mjtracker import interface_to_official_lib
```

### Imports spécifiques

```python
# Plotting
from mjtracker.plotting.batch_plots import batch_merit_profile
from mjtracker.plotting.plots_v2 import plot_merit_profile

# Export
from mjtracker.export import export_compact_json

# Utils
from mjtracker.utils.utils import get_candidates, get_grades
```

## Organisation

- **core/**: Contient les classes et interfaces principales pour manipuler les données de sondage
- **plotting/**: Tout ce qui concerne la génération de graphiques et visualisations
- **legacy/**: Code à maintenir pour compatibilité mais non utilisé en production
- **export/**: Fonctions d'export vers différents formats (JSON, CSV, etc.)
- **utils/**: Fonctions utilitaires réutilisables
- **libs/**: Implémentations des algorithmes de Jugement Majoritaire
- **misc/**: Types, énumérations et autres éléments divers

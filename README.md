# One Look : détection d'inondations embarquée sur images satellites

Modèle d'IA léger qui analyse une tuile Sentinel-1 / Sentinel-2 en une seule passe et décide : **ALERTE**, **NORMAL** ou **INCONNU**.

## Documents

- `docs/cadrage.md` : ce qui mérite une alerte, ce qui n'en mérite pas, l'inconnu, les objectifs.
- `docs/carnet_donnees.md` : sources, licences, composition, découpage, prétraitement, biais.
- `docs/sources.md` : fiches détaillées des jeux de données.
- `docs/notes_exploration.md` : observations visuelles.

## Installation

```
python -m venv .venv
.venv\Scripts\activate.bat        (Windows)  |  source .venv/bin/activate  (Linux / Mac)
python -m pip install -r requirements.txt
```

## Données

Les données ne sont pas versionnées. Pour les reconstruire, suivre la section « Reproduire » de `docs/carnet_donnees.md`.

## Structure

- `src/` : code (statistiques, découpage, prétraitement, jeu de données)
- `scripts/` : téléchargements
- `notebooks/` : exploration
- `docs/` : documentation et figures
- `data/` : données locales (ignorées par Git)

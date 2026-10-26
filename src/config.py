"""Paramètres partagés du projet One Look (fixés dans docs/cadrage.md)."""
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
DATA = RACINE / "data"
S1F11 = DATA / "sen1floods11"
DOCS = RACINE / "docs"

# Règle d'alerte (cadrage, section 3)
VALIDE_MIN_PCT = 50.0     # tuile écartée en dessous
SEUIL_NORMAL_PCT = 0.5    # crue robuste < 0,5 %  -> NORMAL
SEUIL_ALERTE_PCT = 2.0    # crue robuste >= 2 %   -> ALERTE ; entre les deux : zone grise

# Découpage géographique (étape 2.1) : aucun pays dans deux ensembles.
PAYS_TEST = ["Bolivia", "Sri-Lanka", "Somalia"]
PAYS_VAL = ["Spain", "Mekong"]
PAYS_TRAIN = ["USA", "India", "Paraguay", "Ghana", "Nigeria", "Pakistan"]

# Taille de maille (degrés) pour découper les scènes hors Sen1Floods11 par blocs géographiques.
MAILLE_DEG = 10.0

# Bandes
BANDES_S1 = ["VV", "VH"]
BANDES_S2 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]

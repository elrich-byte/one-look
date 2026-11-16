# One Look : détecter ce qui compte, à bord du satellite

Petit modèle d'IA embarquable qui analyse une tuile d'image satellite (5 km × 5 km) **en une seule passe** et décide : **🌊 INONDATION · 🔥 FEU · ✅ NORMAL · ❓ INCONNU**.

| Modèle retenu (test sur pays jamais vus) | |
|---|---|
| Fausses alertes sur images normales | **5 %** |
| Alertes inondation justes (précision) | **95 %** (60 % des crues trouvées) |
| Images inconnues signalées | **100 %** (AUROC 1,00) |
| Taille (INT8) / temps par tuile | **0,21 Mo** / **4,8 ms** sur un cœur CPU |

**Question de recherche : que perd-on avec une simple caméra couleur (RGB) ?** Beaucoup pour l'eau : le même modèle passe de F1 0,61 à 0,31. YOLO (0,54) et CLIP (0,48) ne comblent pas l'écart. → `docs/tableau_comparatif.md`

## Livrables

| | |
|---|---|
| Compte rendu écrit | [`docs/rapport.md`](docs/rapport.md) |
| Présentation (HTML, flèches pour naviguer) | [`docs/presentation/index.html`](docs/presentation/index.html) |
| Vidéo de démonstration | [`docs/demo_one_look.mp4`](docs/demo_one_look.mp4) |
| Carnet de données (sources, licences, biais) | [`docs/carnet_donnees.md`](docs/carnet_donnees.md) |
| Cadrage (ce qui mérite une alerte) | [`docs/cadrage.md`](docs/cadrage.md) |
| Tableau comparatif des modèles | [`docs/tableau_comparatif.md`](docs/tableau_comparatif.md) |
| Modèles embarquables (ONNX INT8) | `modeles/*.int8.onnx` |

## Installation

```
python -m venv .venv
.venv\Scripts\activate.bat        (Windows)   |   source .venv/bin/activate   (Linux / Mac)
python -m pip install -r requirements.txt
```

## Analyser une tuile jamais vue

```
python src/predict.py --s2 tuile_S2.tif --s1 tuile_S1.tif          # modèle complet
python src/predict.py --s2 tuile_S2.tif --modele cnn_rgb           # modèle RGB
python src/demo.py                                                  # démo dans le navigateur
```

Entrée attendue : GeoTIFF 512 × 512 ; Sentinel-2 L1C en 13 bandes (réflectance × 10 000) ; Sentinel-1 en 2 bandes (VV, VH) en dB, facultatif. Le code de décision (`src/decision.py`) n'utilise qu'ONNX Runtime : c'est ce qui tournerait à bord.

## Reproduire

Les données ne sont pas versionnées (plusieurs Go). Toute la chaîne est décrite dans [`docs/rapport.md`](docs/rapport.md), section « Reproduire ».

## Structure

- `src/` : données (statistiques, découpage, téléchargements, prétraitement, cache), modèles (entraînement, évaluation, YOLO, CLIP), décision embarquée, démo
- `scripts/` : téléchargement de l'eau permanente (gsutil)
- `docs/` : documents, figures, résultats (`resultats_*.json`)
- `modeles/` : modèles allégés
- `notebooks/` : exploration
- `data/` : données locales (ignorées par Git)

## Données et licences

Sen1Floods11 (Bonafilia et al., 2020), JRC Global Surface Water (Pekel et al., 2016), Copernicus Sentinel-1 et Sentinel-2 (« Contains modified Copernicus Sentinel data 2016-2021 »), CaBuAr (Rege Cambrin et al., 2023, CDLA-Permissive-2.0). Détails dans le carnet de données.

Code et documentation écrits avec l'aide d'un assistant IA (Claude) ; choix et vérifications validés par l'auteur.

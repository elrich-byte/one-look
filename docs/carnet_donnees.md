# Carnet de données

Ce carnet décrit les données du projet One Look : provenance, licences, composition, découpage, prétraitement, biais et limites. Les chiffres sont produits par les scripts de `src/` et se reproduisent avec les commandes de la dernière section.

## 1. Sources et licences

| Source | Contenu | Rôle | Licence / conditions | Statut |
|---|---|---|---|---|
| **Sen1Floods11 HandLabeled** | 446 tuiles S1 + S2 L1C + masque d'eau annoté à la main, 11 inondations (2016-2019) | crues (ALERTE) et tuiles peu touchées (NORMAL) | non précisée par les auteurs ; citation obligatoire de Bonafilia et al., 2020 | téléchargé |
| **JRC Global Surface Water** (inclus dans Sen1Floods11) | eau permanente | séparer l'eau de crue de l'eau normale | Copernicus, gratuit et sans restriction d'usage ; citer Pekel et al., *Nature*, 2016 ; mention « Source: EC JRC/Google » sur les cartes | téléchargé |
| **Sen1Floods11 perm_water** | 814 tuiles d'eau permanente dans le monde, hors crue, **radar seul** | évaluation des fausses alertes en mode radar seul (pas d'entraînement) | comme Sen1Floods11 | téléchargé et converti (`src/convertir_perm_water.py`) |
| **Copernicus Sentinel-1 GRD et Sentinel-2 L1C** via Copernicus Data Space Ecosystem (openEO) | 61 scènes NORMAL prévues (39 hors crue sur les lieux des crues, 22 négatifs difficiles) et 7 scènes inconnues | NORMAL, négatifs difficiles, INCONNU | licence des données Sentinel Copernicus : gratuite, complète et ouverte ; mention « Contains modified Copernicus Sentinel data [année] » | 62 / 68 téléchargées (`src/scenes_openeo.py`) |
| **CaBuAr** | zones brûlées en Californie, Sentinel-2 seul | INCONNU (phénomène jamais vu) | CDLA-Permissive-2.0 ; citer Rege Cambrin et al., 2023 | script prêt (`src/exporter_cabuar.py`) |
| **Entrées synthétiques** | 7 types d'entrées corrompues ou absurdes, générées à partir de tuiles de test | INCONNU | produites par le projet | `src/inconnu.py` |

Détails des sources dans `docs/sources.md`.

## 2. Format commun

Toutes les tuiles sont ramenées au format de Sen1Floods11 : **512 × 512 pixels, EPSG:4326, environ 10 m par pixel**.

| Capteur | Bandes | Unité | Pas de donnée |
|---|---|---|---|
| Sentinel-1 | VV, VH | dB (sigma0) | NaN |
| Sentinel-2 L1C | B1 à B12, dont B8A et B10 (13) | réflectance × 10 000 | 0 sur toutes les bandes |

Les scènes Copernicus sont prises **avant 2022** : depuis le 25/01/2022, les produits S2 L1C ont un décalage de +1000 (baseline 04.00) qui les rendrait incomparables. Le radar d'une scène vient d'**une seule acquisition** (fenêtre de 12 jours) : une moyenne de plusieurs dates lisserait le chatoiement et donnerait au modèle un indice trompeur (« image lisse = normale »).

## 3. Étiquettes

Règle fixée dans `docs/cadrage.md`, appliquée par `src/decoupage.py` :

- **crue robuste** = eau du masque − eau permanente JRC élargie de 3 pixels, sans les taches de moins de 50 pixels ;
- moins de 50 % de pixels valides → **ECARTEE** ;
- crue < 0,5 % → **NORMAL** ; 0,5-2 % → **GRIS** (exclue de l'entraînement, analysée à part) ; ≥ 2 % → **ALERTE** ;
- scènes hors Sen1Floods11 : NORMAL par construction (lieu et date choisis hors crue), ou INCONNU.

Les scènes hors crue ne sont pas annotées pixel par pixel : leur étiquette repose sur le choix de la date (saison sèche ou basses eaux), à vérifier visuellement après téléchargement.

## 4. Découpage entraînement / validation / test

Le découpage officiel de Sen1Floods11 tire les tuiles au hasard **à l'intérieur de chaque pays** : des tuiles voisines d'une même scène se retrouvent en entraînement et en test, ce qui gonfle les scores. Il n'est pas utilisé.

Découpage du projet (`src/config.py`), sans aucun pays commun entre deux ensembles :

| Ensemble | Pays Sen1Floods11 | Rôle |
|---|---|---|
| train | USA, Inde, Paraguay, Ghana, Nigeria, Pakistan | apprentissage |
| val | Espagne, Mékong (Cambodge) | réglage des hyperparamètres et du seuil de décision |
| test | Bolivie, Sri Lanka, Somalie | évaluation finale « zones jamais vues » ; une seule fois, à la fin |

Chaque ensemble de test a un « voisin » dans l'entraînement (Bolivie / Paraguay, Sri Lanka / Inde, Somalie / Ghana-Nigeria), mais jamais la même scène.

- Scènes hors crue sur les mêmes lieux : même ensemble que leur pays.
- Scènes mondiales (eau permanente, négatifs difficiles) : par **blocs de 10° × 10°** (70 / 15 / 15 %, tirage déterministe), pour que deux scènes proches restent dans le même ensemble.
- Scènes inconnues : ensemble à part, jamais vu pendant l'entraînement ni le réglage.

### Composition actuelle (Sen1Floods11 seul)

| Ensemble | ALERTE | NORMAL | GRIS | ECARTEE | Total |
|---|---|---|---|---|---|
| train | 114 | 102 | 53 | 34 | 303 |
| val | 29 | 14 | 17 | 0 | 60 |
| test | 35 | 30 | 14 | 4 | 83 |
| **Total** | **178** | **146** | **84** | **38** | **446** |

Tuiles utilisables pour l'apprentissage et l'évaluation (ALERTE + NORMAL) : 216 train, 43 val, 65 test.

### Composition au 26/10/2026 (`data/index.csv`)

| Ensemble | ALERTE | NORMAL | INCONNU | Rôle |
|---|---|---|---|---|
| train | 114 | 141 (102 Sen1Floods11 + 39 Copernicus) | | apprentissage |
| val | 29 | 21 (14 + 7) | | réglage |
| test | 35 | 41 (30 + 11) | | évaluation finale |
| eval_eau_permanente | | 814 (perm_water, radar seul) | | fausses alertes sur eau normale |
| inconnu | | | 5 scènes Copernicus | détection de l'inconnu |

S'y ajoutent les 7 types d'entrées synthétiques (`src/inconnu.py`), et CaBuAr en option. Il manque 6 scènes Copernicus (4 en Inde, banquise du Groenland, glacier d'Islande) et le S2 du salar d'Uyuni : le script les retente à chaque relance.

### Constats sur les données ajoutées

- **perm_water n'est pas au format annoncé** : le radar est ramené entre 0 et 1 par une transformation non documentée, il n'y a pas de Sentinel-2, et les tuiles ne font pas 512 × 512. Une conversion en dB a été estimée en calant les médianes « terre » et « eau permanente » sur Sen1Floods11 (VV ≈ 57,9·x − 55,0 ; VH ≈ 45,5·x − 46,4), avec une précision de 2 à 3 dB. Trop imprécis pour l'entraînement : ces tuiles servent seulement à mesurer les fausses alertes en mode radar seul.
- **Écart radar entre sources** : sur les tuiles normales, la médiane VV vaut −10,0 dB pour Sen1Floods11 (Google Earth Engine) et −11,5 dB pour les scènes Copernicus (openEO, sigma0 ellipsoïde). Écart de 1,5 dB, en partie dû aux paysages différents. À surveiller : si le modèle sépare les sources au lieu des classes, c'est un biais.
- **Pixels optiques négatifs isolés** (moins de 0,003 % d'une tuile) dans 5 scènes Copernicus : artefacts du serveur, ramenés à 0 par la normalisation. Le contrôle de format tolère jusqu'à 1 % de valeurs hors plage.

## 5. Prétraitement

Module unique `src/donnees.py`, utilisé tel quel par l'entraînement, l'évaluation et la démo.

1. **Contrôles de format** (`verifier`) : nombre de bandes, taille, S1 bien en dB, S2 dans la plage de réflectance. En cas d'échec : décision INCONNU, sans passer par le modèle.
2. **Disponibilité des capteurs** : un capteur est lisible si au moins 20 % de ses pixels sont valides. Si aucun ne l'est : INCONNU.
3. **Normalisation** : S1 écrêté à [-50, 5] dB ; S2 divisé par 10 000 et écrêté à [0, 2] ; puis standardisation par canal avec moyenne et écart-type calculés **sur l'ensemble train uniquement** (`data/normalisation.json`, 303 tuiles Sen1Floods11 + 39 scènes Copernicus).
4. **Valeurs manquantes** : pixels NaN ou vides et capteur absent remplacés par 0 (la moyenne après standardisation).
5. Sortie : tableau de **15 canaux** (VV, VH, B1…B12) de 512 × 512, et un indicateur de disponibilité (S1, S2).

`src/dataset.py` ajoute le *modality dropout* (un capteur masqué au hasard à l'entraînement) et la réduction de taille (ex. 128 × 128) pour un modèle léger.

Vérification : sur les 446 tuiles, 443 passent le prétraitement ; les 3 rejetées sont des tuiles vides déjà classées ECARTEE. Sur les 7 types d'entrées synthétiques, les 3 formats invalides sont bien rejetés et les 4 contenus absurdes passent le contrôle, comme prévu : c'est au modèle de les signaler (étape 4).

## 6. Biais et limites

- **Peu d'événements** : 11 inondations, entre 2016 et 2019. Le modèle risque d'apprendre leurs paysages plutôt que la crue.
- **Déséquilibre géographique** : 69 tuiles pour les USA, 15 pour la Bolivie ; aucune tuile d'Europe du Nord, d'Asie de l'Est ou d'Océanie dans les crues.
- **Normales prises pendant les crues** dans Sen1Floods11 : corrigé en partie par les scènes hors crue et perm_water.
- **Définition de la crue dépendante du JRC**, imparfait sur les rivières étroites et les zones inondées chaque saison (rizières, Tonlé Sap).
- **Annotations faites à partir de S2** : sous les nuages, pas d'étiquette, même si le radar voit l'eau.
- **Différences de traitement** : le radar de Sen1Floods11 vient de Google Earth Engine, celui des scènes ajoutées d'openEO (sigma0 ellipsoïde). Écart possible de quelques dB, à contrôler : si le modèle sépare les sources au lieu des classes, c'est un biais de source.
- **Scènes hors crue étiquetées par la date**, sans annotation pixel par pixel.
- **Heuristique de nuages grossière** (B2 et B4 > 0,2), utilisée pour décrire, pas pour filtrer.
- **Licence de Sen1Floods11 non précisée** par les auteurs.

## 7. Reproduire

```
python src/stats_tuiles.py              # statistiques par tuile -> docs/stats_tuiles.csv
python src/decoupage.py                 # classes et découpage par pays
scripts\telecharger_perm_water.bat      # eau permanente (gsutil)
python src/scenes_openeo.py             # scènes Copernicus (compte gratuit, openeo)
python src/exporter_cabuar.py --n 15    # zones brûlées inconnues (datasets, hdf5plugin)
python src/convertir_perm_water.py      # perm_water en dB, 512 x 512 (évaluation seule)
python src/construire_index.py          # index unique -> data/index.csv
python src/donnees.py --stats           # normalisation (train uniquement)
python src/donnees.py --verifier        # contrôle de toutes les tuiles
python src/inconnu.py                   # contrôle des entrées synthétiques
```

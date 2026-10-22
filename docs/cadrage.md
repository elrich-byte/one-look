# Cadrage du projet One Look

Ce document fixe ce qui mérite une alerte, ce qui n'en mérite pas, et ce que le modèle doit signaler comme inconnu. Les chiffres viennent de `docs/stats_tuiles.csv`, produit par `src/stats_tuiles.py` sur les 446 tuiles annotées à la main de Sen1Floods11.

## 1. Scénario

Le modèle tourne **à bord du satellite**, sur l'ordinateur de bord. Pour chaque tuile acquise, il répond en une seule passe :

| Décision | Ce que fait le satellite |
|---|---|
| **ALERTE** | envoie en priorité un message léger (position, type, confiance) et la tuile |
| **NORMAL** | ne transmet pas la tuile, ou plus tard en basse priorité : c'est là que la bande passante est économisée |
| **INCONNU** | transmet la tuile pour qu'un analyste au sol la regarde |

Au sol, des analystes (par exemple le service de cartographie d'urgence Copernicus) vérifient l'alerte. Dans ce projet, un ordinateur en CPU seul joue le rôle de l'ordinateur de bord : c'est pourquoi la taille du modèle et le temps d'inférence sur CPU font partie des critères.

Référence : le satellite Φ-sat-1 de l'ESA (2020) embarquait déjà une puce d'IA pour écarter à bord les images nuageuses.

## 2. Anomalie visée : les inondations

- **Données au bon format** : Sen1Floods11 fournit des paires Sentinel-1 / Sentinel-2 co-registrées, annotées à la main, sur 11 événements et 4 continents.
- **Radar** : Sentinel-1 voit à travers les nuages, presque toujours présents pendant une crue. Sur les tuiles du Pakistan, l'heuristique de nuages couvre en moyenne 38 % de l'image S2.
- **Signature physique nette** : l'eau calme est très sombre en radar et sombre dans l'infrarouge en optique.
- **Périmètre tenable en trois semaines.**

Les zones brûlées (CaBuAr) restent une option si le temps le permet, sans engagement : une seule région (Californie), pas de radar, et un niveau de traitement probablement différent (L2A contre L1C).

## 3. Règle d'alerte

### Définition de l'eau de crue

Le masque annoté contient **toute** l'eau du jour, rivières et lacs compris. L'eau de crue est l'eau du masque qui n'est pas de l'eau permanente :

```
crue = eau du masque  −  eau permanente (JRC)
```

L'exploration a montré que le JRC (30 m de résolution) manque les rivières étroites et les petites mares, que cette formule compte alors à tort comme crue. On utilise donc une version **robuste** :

- l'eau permanente est élargie de **3 pixels** (environ 30 m) ;
- les taches de crue de moins de **50 pixels** (environ 0,5 ha) sont ignorées.

Effet mesuré : 26 tuiles passent sous le seuil d'alerte avec la version robuste, essentiellement des bords de rivières.

### Seuils

Le pourcentage est calculé sur les **pixels valides** (masque différent de -1).

| Classe | Règle | Tuiles |
|---|---|---|
| Écartée | moins de 50 % de pixels valides | 38 |
| **NORMAL** | crue robuste **< 0,5 %** | 146 |
| Zone grise | crue robuste entre **0,5 % et 2 %** | 84 |
| **ALERTE** | crue robuste **≥ 2 %** | 178 |

Une tuile de 512 × 512 pixels à 10 m couvre environ 26 km² : 2 % représentent environ **0,5 km² inondés**.

**Pourquoi ces valeurs.** L'histogramme (`docs/figures/hist_crue.png`) est très concentré près de 0 : la médiane de crue robuste est de 1,2 %. Un seuil d'alerte à 2 % sépare les vraies crues des débordements marginaux. La zone grise (0,5-2 %) contient les cas ambigus (petites surfaces, erreurs du JRC) : elle est **retirée de l'entraînement** pour ne pas apprendre sur des étiquettes floues, mais **gardée en test** et analysée à part. Avec un seuil bas à 0,2 %, la zone grise aurait contenu 133 tuiles : trop de données perdues.

### Répartition par pays (tuiles valides)

| Pays | NORMAL | Zone grise | ALERTE |
|---|---|---|---|
| Bolivie | 3 | 2 | 8 |
| Ghana | 18 | 7 | 10 |
| Inde | 10 | 12 | 43 |
| Mékong | 5 | 8 | 17 |
| Nigeria | 6 | 2 | 9 |
| Pakistan | 9 | 4 | 7 |
| Paraguay | 25 | 14 | 26 |
| Somalie | 11 | 1 | 13 |
| Espagne | 9 | 9 | 12 |
| Sri Lanka | 16 | 11 | 14 |
| USA | 34 | 14 | 19 |
| **Total** | **146** | **84** | **178** |

**Limite importante** : les 146 tuiles NORMAL sont prises **pendant** une crue, dans la zone touchée. Aucune ne représente un paysage vraiment normal. Des scènes hors crue seront ajoutées à l'étape 2.

## 4. Ce qui ne doit pas déclencher d'alerte (négatifs difficiles)

| Piège | Pourquoi il trompe | Observé ? | Où trouver des exemples |
|---|---|---|---|
| Rivières, lacs, réservoirs normaux | c'est de l'eau | oui (USA_770353, Paraguay_683296) | tuiles Sen1Floods11 à forte eau permanente ; Copernicus Browser |
| Rivières étroites absentes du JRC | comptées comme crue par la formule | oui (Paraguay_1056717) | Sen1Floods11, corrigé par la crue robuste |
| Parcelles agricoles, sols nus ou labourés | très sombres en radar | oui (USA_181503, Paraguay_44682) | mêmes régions hors saison de crue |
| Sols arides | zones sombres en radar | oui (Pakistan_760290) | Copernicus Browser, déserts |
| Nuages | masquent tout en optique | oui (Pakistan_474121, India_952728) | CloudSEN12 |
| Ombres de nuages | sombres en optique, ressemblent à de l'eau | oui (Paraguay_683296) | CloudSEN12 |
| Ombres de relief | sombres en radar | non observé | Copernicus Browser, zones montagneuses |
| Rizières inondées en saison | eau réelle mais normale | non observé | Copernicus Browser, Asie du Sud-Est |
| Neige, glace | signature proche de l'eau en optique | non observé | Copernicus Browser, Alpes en hiver |
| Routes, pistes, sable sec | surfaces lisses, sombres en radar | non observé | Copernicus Browser, aéroports, déserts |

Objectif de l'étape 2 : au moins autant de tuiles normales que de tuiles d'alerte, dont une part importante de ces négatifs difficiles.

## 5. Ce qui doit être signalé comme INCONNU

| Cas | Exemple | Détection |
|---|---|---|
| Entrée inexploitable | mauvais nombre de bandes, valeurs aberrantes, fichier corrompu | vérifications de format **avant** le modèle |
| Aucun capteur lisible | S2 entièrement nuageux **et** S1 absent | part de pixels valides par capteur |
| Tuile majoritairement vide | bord de fauchée (ex. USA_347609) | moins de 50 % de pixels valides |
| Confiance trop basse | tuile ambiguë | seuil sur la confiance du modèle (réglé à l'étape 4) |
| Paysage ou phénomène jamais vu | banquise, désert extrême, séisme, glissement de terrain | score hors distribution (energy score, distance dans l'espace des features) |

Images de test pour l'inconnu : événements du Copernicus Emergency Management Service (autres catastrophes) avec leurs images Sentinel, et quelques scènes de paysages absents de Sen1Floods11. Les images xBD (haute résolution, RGB) ne servent qu'au cas « autre capteur ».

## 6. Capteur : un modèle qui accepte S1, S2 ou les deux

Sentinel-1 et Sentinel-2 sont **deux satellites différents**. Un modèle embarqué sur l'un ne voit que son propre capteur. Le choix retenu est donc **un seul modèle qui accepte S1 seul, S2 seul ou les deux**, entraîné avec du *modality dropout* (un capteur masqué au hasard pendant l'entraînement).

Justification par les données :
- S2 est souvent gêné par les nuages : 124 tuiles sur 446 ont plus de 10 % de nuages (heuristique), 39 en ont plus de 30 %.
- S1 est le plus souvent complet : seules 45 tuiles ont des NaN radar, 8 en ont plus de 50 %.
- Mais le radar seul confond l'eau avec les champs et sols sombres : l'optique aide quand elle est dégagée.

| Situation | Modèle S1 + S2 | S1 seul | S2 seul |
|---|---|---|---|
| S2 dégagé | décide | décide | décide |
| S2 nuageux | décide grâce au radar | décide | INCONNU |
| S1 manquant | décide grâce à S2 | INCONNU | décide |

La version S1 + S2 sert de référence de performance ; les versions mono-capteur montrent ce qui tournerait réellement à bord.

## 7. Évaluation honnête

Le découpage officiel train / valid / test de Sen1Floods11 mélange les mêmes pays (tirage aléatoire par tuile) : des tuiles voisines d'une même scène peuvent se retrouver en entraînement et en test, ce qui gonfle les scores. Seule la Bolivie est tenue à l'écart.

Décision : **découpage par pays** (fait à l'étape 2). La Bolivie et au moins un autre pays sont réservés au test « zones jamais vues » ; les autres pays sont répartis entre entraînement et validation, sans qu'un même pays apparaisse dans deux ensembles.

## 8. Objectifs chiffrés

| Critère | Cible indicative |
|---|---|
| Taux de fausses alertes sur images normales | ≤ 5 % |
| Rappel sur les vraies crues (classe ALERTE) | ≥ 85 % |
| Précision des alertes | ≥ 90 % |
| Détection de l'inconnu (AUROC) | ≥ 0,85 |
| Taille du modèle | < 20 Mo (idéalement < 5 Mo en INT8) |
| Inférence par tuile, CPU | < 100 ms |

En cas d'arbitrage, **la précision passe avant le rappel** : une fausse alerte gaspille de la bande passante et décrédibilise le système. Le seuil de décision sera réglé d'abord pour tenir la cible de fausses alertes.

## 9. Limites connues

- Les « normales » de Sen1Floods11 sont prises pendant des crues : il manque de vrais paysages normaux.
- Fort déséquilibre entre pays (69 tuiles USA, 15 Bolivie) ; 11 événements seulement.
- La définition de la crue dépend du JRC, imparfait sur les petits cours d'eau et les zones inondées chaque saison.
- Annotations faites sur S2 : sous les nuages, pas d'étiquette, même si le radar voit l'eau.
- L'heuristique de nuages (B2 et B4 > 0,2) est grossière ; elle sert à décrire les données, pas à les filtrer.
- Licence des données Sen1Floods11 non précisée par les auteurs.
- Inondations sous forêt dense mal vues par le radar en bande C.

## Reproduire les chiffres

```
python src/stats_tuiles.py
```

Produit `docs/stats_tuiles.csv` (une ligne par tuile) et `docs/figures/hist_crue.png`.

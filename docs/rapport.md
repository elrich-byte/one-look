# One Look : compte rendu

**Détecter les inondations et les feux de forêt à bord d'un satellite, en une seule passe et sans fausses alertes**

Projet individuel, octobre-novembre 2026. Dépôt : https://github.com/elrich-byte/one-look

---

## 1. Résumé

Le projet construit un modèle d'IA assez petit pour tourner à bord d'un satellite. Il analyse une tuile d'image de 5 km × 5 km et répond **INONDATION**, **FEU**, **NORMAL** ou **INCONNU**.

Le modèle retenu, un petit réseau convolutif sur les 13 bandes de Sentinel-2 et le radar Sentinel-1, pèse **0,21 Mo** une fois compressé en INT8. Il analyse une tuile en **4,8 ms sur un seul cœur CPU**. Sur des pays jamais vus à l'entraînement, il atteint **5 % de fausses alertes**, et **95 % de ses alertes inondation sont justes**, pour 60 % des inondations trouvées. Il signale **100 % des images inconnues testées** (lave, glacier, mine, salar, images corrompues), tout en gardant 88 % des images connues.

La question de recherche, posée après le point avec l'encadrant, était : *que perd-on avec une simple caméra couleur (RGB) ?* Réponse : beaucoup pour l'eau. Le même modèle en RGB tombe de F1 0,61 à 0,31. Un modèle pré-entraîné (MobileNet, F1 0,46), YOLO (F1 0,54) ou CLIP sans entraînement (F1 0,48, 51 % de fausses alertes) ne comblent pas l'écart. L'information utile pour l'eau se trouve dans l'infrarouge.

## 2. Cadrage

### 2.1 Scénario
À bord, un ordinateur reçoit chaque tuile, la contrôle, la classe et décide :
- **alerte** : message prioritaire et tuile envoyée au sol ;
- **normal** : tuile non transmise, ou plus tard ;
- **inconnu** : tuile envoyée à un analyste.

Sentinel-1 (radar) et Sentinel-2 (optique) sont deux satellites différents : un modèle embarqué ne voit qu'un capteur. Le modèle complet est donc entraîné avec du *modality dropout* (un capteur masqué au hasard), pour accepter l'un, l'autre ou les deux.

### 2.2 Ce qui mérite une alerte
- **Inondation** : eau du jour (masque annoté) moins l'eau permanente (JRC Global Surface Water élargie de 30 m, taches de moins de 0,5 ha ignorées). Alerte si au moins **2 %** de la tuile est en crue (environ 0,5 km²). Normale sous **0,5 %**. Entre les deux, zone grise exclue de l'apprentissage.
- **Feu** : zone brûlée visible, quelques semaines après l'incendie.

### 2.3 Ce qui ne doit pas alerter
Rivières, lacs et réservoirs, rizières, bassins salants, ombres de nuages, champs labourés et sols nus (sombres au radar), neige, désert, aéroports, villes. Ces pièges ont été observés dans les données ou ajoutés exprès (22 scènes Copernicus).

### 2.4 Ce qui doit être signalé comme inconnu
Entrées au mauvais format (rejetées par des contrôles avant le modèle), tuiles vides, et paysages ou phénomènes jamais vus : coulée de lave, glacier, mine à ciel ouvert, salar. Sept types d'entrées synthétiques complètent ce jeu : bruit, image constante, nuage total sans radar, bandes mélangées, bandes manquantes, radar non converti en dB, aucun capteur.

Détails : `docs/cadrage.md`.

## 3. Données

| Source | Contenu | Rôle | Licence |
|---|---|---|---|
| Sen1Floods11 HandLabeled | 446 tuiles S1 + S2 L1C, masques d'eau annotés, 11 inondations | crues et normales | non précisée ; citation de Bonafilia et al., 2020 |
| JRC Global Surface Water | eau permanente | définition de la crue | Copernicus, libre ; citation de Pekel et al., 2016 |
| Copernicus (openEO) | 76 scènes S1 + S2 : saison sèche, pièges, feux, inconnu | normales, pièges, feux de test, inconnu | données Sentinel libres et ouvertes |
| CaBuAr | 80 zones brûlées et 61 images « avant », Californie | feux (apprentissage) | CDLA-Permissive-2.0 ; Rege Cambrin et al., 2023 |
| Sen1Floods11 perm_water | 814 tuiles d'eau permanente, radar seul | test de fausses alertes | comme Sen1Floods11 |

**Format commun** : 512 × 512 px, environ 10 m, radar en dB, 13 bandes optiques en réflectance × 10 000. Pour les modèles, les tuiles sont réduites à 128 × 128 (1 pixel sur 4, environ 40 m).

**Découpage sans fuite.** Le découpage officiel de Sen1Floods11 tire les tuiles au hasard *dans* chaque pays, ce qui gonfle les scores : il n'est pas utilisé. Aucun pays n'apparaît dans deux ensembles.

| Ensemble | Inondation | Feu | Normal | Pays |
|---|---|---|---|---|
| Apprentissage | 114 | 55 | 189 | USA, Inde, Paraguay, Ghana, Nigeria, Pakistan, Californie |
| Validation | 29 | 27 | 38 | Espagne, Mékong, Portugal, un pli de la Californie |
| Test | 35 | 4 | 41 | Bolivie, Sri Lanka, Somalie, Grèce, Sardaigne, Australie |

**Corrections apportées en cours de route** (toutes documentées dans `docs/carnet_donnees.md`) :
- *perm_water* n'était pas au format annoncé (radar ramené entre 0 et 1) : conversion approximative en dB, usage limité au test.
- Sur 10 feux téléchargés, 4 ne montraient aucune brûlure (vérification visuelle, `docs/figures/feux_copernicus.png`) : écartés.
- CaBuAr est en L2A (atmosphère corrigée), plus sombre que nos images L1C : harmonisation par bande, pour que le modèle n'apprenne pas « image sombre = feu ».
- Les téléchargements Copernicus d'avant 2022 évitent le décalage de +1000 introduit en janvier 2022.

## 4. Exploration : ce que voit Sentinel-2

Sur 444 000 pixels annotés, la capacité à séparer l'eau de la terre (AUC) vaut **0,98** pour l'indice MNDWI (vert + infrarouge moyen), 0,94 pour le proche infrarouge seul, 0,89 pour le radar VH… et **0,69** seulement pour la meilleure bande RGB. Dans le visible, l'eau de crue boueuse est même un peu plus claire que la végétation.

Une règle simple (détecteur d'eau par pixel puis seuil sur la part d'eau) donne sur la validation **100 % de fausses alertes en RGB**, contre 38 % avec l'image complète. Les fausses alertes restantes viennent des pièges attendus : ombres de nuages, rizières, bassins salants. Détails : `docs/resultats_baseline.md`.

## 5. Modèles et résultats

### 5.1 Modèles comparés
- **Petit CNN** (4 blocs convolutifs, 186 000 paramètres), appris de zéro, en RGB puis sur l'image complète (15 canaux).
- **MobileNetV3-small** pré-entraîné sur ImageNet (photos), en RGB.
- **YOLOv8n**, détecteur de rectangles, en RGB : les rectangles d'apprentissage sont fabriqués automatiquement à partir des masques ; la décision pour la tuile est le rectangle le plus sûr.
- **CLIP** (Hugging Face, vision-langage) sans aucun entraînement : l'image est comparée à des phrases (« une image satellite d'une zone inondée »…).

Apprentissage sur CPU : augmentation par retournements et rotations, poids de classes, choix du meilleur état sur la validation. Un **seuil d'alerte** est réglé sur la validation : le modèle n'alerte que si la classe d'alerte est assez probable.

### 5.2 Résultats sur le test (pays jamais vus)

| Modèle | F1 macro | Inondation P / R | Feu P / R | Fausses alertes | Taille | ms / tuile |
|---|---|---|---|---|---|---|
| Petit CNN — RGB | 0,31 | 67 % / 17 % | 0 % / 0 % | 12 % | 0,2 Mo | 3,3 |
| MobileNetV3 — RGB | 0,46 | 50 % / 20 % | 40 % / 50 % | 20 % | 6,1 Mo (1,9 en INT8) | 1,3 |
| YOLOv8n — RGB | 0,54 | 64 % / 46 % | 40 % / 50 % | 29 % | 12,6 Mo (32 bits) | 40 |
| CLIP sans entraînement — RGB | 0,48 | 61 % / 57 % | 20 % / 75 % | 51 % | 605 Mo | 148 (multi-cœurs) |
| **Petit CNN — complet** | **0,61** | **95 % / 60 %** | 33 % / 25 % | **5 %** | **0,2 Mo** | **4,8** |

P = précision, R = rappel. Test : 35 inondations, 4 feux, 41 normales. Temps : ONNX Runtime, un seul cœur CPU. Tailles en INT8 pour les petits CNN.

**YOLO est le meilleur modèle RGB** (F1 0,54 contre 0,31 pour le petit CNN RGB) : localiser la *forme* de la zone inondée ou brûlée compense en partie l'absence d'infrarouge. Il reste derrière le modèle complet, avec 6 fois plus de fausses alertes et un modèle 60 fois plus lourd.

**Effet du seuil d'alerte** (modèle complet) : sans seuil, 56 % de fausses alertes et F1 0,51 ; avec le seuil, 5 % de fausses alertes et F1 0,61. C'est le levier le plus fort pour respecter « ne pas alerter pour tout et n'importe quoi ».

### 5.3 Un seul capteur
| Entrée du modèle complet | F1 | Inondation P / R | Fausses alertes |
|---|---|---|---|
| Radar + optique | 0,61 | 95 % / 60 % | 5 % |
| Optique seule | 0,59 | 100 % / 37 % | 2 % |
| Radar seul | 0,23 | — / 0 % | 0 % |

En radar seul, le modèle n'est jamais assez sûr pour alerter. Les 0 % de fausses alertes sur les 814 tuiles d'eau permanente (contre 54 % sans seuil) ne démontrent donc rien sur sa qualité : un modèle dédié au radar, avec son propre seuil, serait nécessaire.

### 5.4 Détection de l'inconnu
Le score « énergie », essayé d'abord, échoue (AUROC 0,22) : le bruit aléatoire rend le modèle *plus* sûr de lui. La **distance de Mahalanobis** dans l'espace des caractéristiques internes, avec un seuil qui garde 95 % des images connues de validation, atteint **AUROC 1,00** pour le modèle complet. Elle signale 100 % des scènes inconnues et des entrées synthétiques, et garde 88 % des images connues du test. En RGB, elle est moins efficace : AUROC 0,82 pour le petit CNN, 0,58 pour MobileNet.

### 5.5 Embarquabilité
Export ONNX puis quantification INT8 statique (calibration sur 100 tuiles d'apprentissage) :
- petit CNN complet : 0,76 → **0,21 Mo**, 7,9 → **4,8 ms**, F1 inchangé (0,51 → 0,52 sans seuil) ;
- MobileNetV3 : 6,1 → 1,9 Mo, mais F1 de 0,45 à 0,22 et pas plus rapide. Ses convolutions séparables supportent mal la quantification simple.

## 6. Analyse des échecs
- **Inondations manquées** (13 sur 35 pour le modèle complet) : crues diffuses ou en bordure, surtout au Sri Lanka. C'est le prix du seuil d'alerte.
- **Fausses alertes sans seuil** : surtout des tuiles « normales » prises pendant une crue. Elles contiennent de l'eau que la règle ne compte pas comme crue (rivières larges, eau permanente mal cartographiée). La frontière crue / eau normale est floue même pour l'étiquette.
- **Feux** : 4 seulement en test (3 manqués par le modèle complet). Les images CaBuAr « avant incendie » sont souvent prises pour des feux en validation : la végétation sèche californienne ressemble à du brûlé. Trop peu de feux variés pour conclure.
- **RGB** : sans infrarouge, le petit CNN confond eau, ombre et sol sombre ; avec le seuil, il préfère ne rien dire (17 % des crues trouvées).
- **Radar seul** : aucune alerte (voir 5.3).

## 7. Limites
- 11 événements d'inondation seulement (2016-2019) et 4 feux de test : les incertitudes sont grandes.
- Étiquettes de crue dépendantes de la carte d'eau permanente ; scènes Copernicus étiquetées par leur date, sans annotation pixel par pixel ; certaines assemblent deux dates.
- CaBuAr harmonisé par une correction linéaire simple.
- Licence des données Sen1Floods11 non précisée par les auteurs.
- Tuiles réduites à 40 m pour les modèles : les petites crues disparaissent.

## 8. Reproduire
```
python -m pip install -r requirements.txt
python src/stats_tuiles.py && python src/decoupage.py
scripts\telecharger_perm_water.bat && python src/convertir_perm_water.py
python src/scenes_openeo.py            # compte Copernicus gratuit
python src/exporter_cabuar.py --n 80
python src/construire_index.py && python src/donnees.py --stats
python src/cache.py && python src/harmoniser_cabuar.py
python src/entrainer.py && python src/evaluer.py && python src/evaluer_capteurs.py
python src/yolo.py preparer && python src/yolo.py entrainer && python src/yolo.py evaluer
python src/clip_zero_shot.py && python src/tableau.py
python src/predict.py --s2 tuile_S2.tif --s1 tuile_S1.tif     # une tuile
python src/demo.py                                              # démonstration
```

## 9. Démonstration
- `src/demo.py` : application dans le navigateur. On choisit une tuile de test ou on dépose un GeoTIFF, et l'application affiche l'image en RGB et en infrarouge avec les décisions du modèle RGB et du modèle complet.
- `docs/demo_one_look.mp4` : vidéo de démonstration sur des tuiles jamais vues.
- `src/predict.py` : décision en ligne de commande, avec le code qui tournerait à bord (ONNX Runtime seul, modèle INT8).

## 10. Outils et aide
Python, PyTorch, TorchGeo (chargement), rasterio, scikit-learn, ONNX Runtime, Ultralytics (YOLO), Hugging Face (CLIP, CaBuAr), openEO (Copernicus). Modèles pré-entraînés utilisés : MobileNetV3-small (ImageNet), YOLOv8n (COCO), CLIP ViT-B/32. Le code et la documentation ont été écrits avec l'aide d'un assistant IA (Claude) ; les choix et les vérifications ont été validés par l'auteur.

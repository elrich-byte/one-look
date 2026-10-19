# Sources de données

## Sen1Floods11

* Lien : https://github.com/cloudtostreet/Sen1Floods11
* Licence : non indiquée dans le README du dépôt. **À confirmer** (fichier LICENSE du dépôt, ou contact des auteurs) avant le rendu.
* Article à citer : Bonafilia, D., Tellman, B., Anderson, T., & Issenberg, E. (2020). *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1.* IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 210-211.
* Capteurs et bandes : Sentinel-1 SAR (2 bandes : VV, VH, en dB) et Sentinel-2 MSI (13 bandes : B1 à B12, dont B8A).
* Niveau de traitement S2 : L1C (réflectance au sommet de l'atmosphère, *Top-Of-Atmosphere*), confirmé par le README.
* Taille et résolution des tuiles : 512 × 512 pixels, environ 10 m par pixel.
* Valeurs du masque : -1 (pas de donnée / non valide), 0 (pas d'eau), 1 (eau).
* Nombre d'événements / pays : 11 événements d'inondation, nommés par pays dans les noms de fichiers. Liste exacte à extraire de `Sen1Floods11_Metadata.geojson`.
* Nombre de tuiles annotées à la main : 446 (sous-ensemble HandLabeled), réparties en train / val / test officiels, plus un test séparé sur la Bolivie (`flood_bolivia_data.csv`). Le reste du jeu (environ 4 385 tuiles) est en étiquetage faible (WeaklyLabeled).
* Comment télécharger : bucket Google Cloud public, avec `gsutil` (installé via pip) :
  `gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/<S1Hand|S2Hand|LabelHand|JRCWaterHand> data/sen1floods11/`
  Découpages officiels : `gs://sen1floods11/v1.1/splits/`. Métadonnées : `gs://sen1floods11/v1.1/Sen1Floods11_Metadata.geojson`.

### Format vérifié (exploration du 19/10/2026)

* GeoTIFF 512 × 512, EPSG:4326 (coordonnées en degrés), résolution 0,0000898° soit environ 10 m.
* S1 : 2 bandes (VV, VH) en dB, plage d'environ [-50, 0]. NaN fréquents (zones hors fauchée radar) : jusqu'à 61 % de la tuile sur l'exemple inspecté.
* S2 : 13 bandes, réflectance × 10 000 (entiers, max observé 5662). 0 = probablement pas de donnée (à vérifier sur d'autres tuiles).
* Masque : valeurs -1, 0, 1, conformes à la documentation.
* JRC (eau permanente) : 0 / 1.

### Remarques / limites

* Les nuages ne sont pas masqués en L1C et peuvent fausser la détection optique.
* La végétation dense masque l'eau en radar bande C (inondations sous couvert forestier).
* Les NaN de S1 doivent être traités dans le prétraitement (remplacement et masque des pixels invalides).
* Les pixels -1 du masque doivent être ignorés dans le calcul de la perte (`ignore_index = -1`).
* Toutes les tuiles proviennent de périodes d'inondation : il faudra ajouter de vraies scènes hors crue.

## CaBuAr

* Lien : https://huggingface.co/datasets/DarthReca/california_burned_areas
* Licence : CDLA-Permissive-2.0.
* Article à citer : Rege Cambrin, D., Colomba, L., & Garza, P. (2023). *CaBuAr: California burned areas dataset for delineation [Software and Data Sets].* IEEE Geoscience and Remote Sensing Magazine, 11(3), 106-113. DOI : 10.1109/MGRS.2023.3292467.
* Capteurs et bandes : Sentinel-2 MSI, 12 canaux, en deux dates (pré-incendie et post-incendie).
* Niveau de traitement S2 : **à confirmer** (non indiqué sur la page ; probablement L2A, puisqu'il y a 12 bandes et non 13).
* Taille et résolution des tuiles : 512 × 512 pixels ; résolution **à confirmer** (non indiquée sur la page).
* Format des fichiers : HDF5, compressé avec BZip2 ; nécessite `h5py` et `hdf5plugin`. Métadonnées (coordonnées, CRS, dates) dans `metadata.parquet`.
* Valeurs du masque : binaire, brûlé / non brûlé (vérité terrain : California Department of Forestry and Fire Protection, CAL FIRE). Valeurs exactes à vérifier à l'ouverture.
* Zone géographique et période : Californie (États-Unis). Nombre d'événements et période **à confirmer** dans l'article.
* Découpage : 5 partitions aléatoires prédéfinies (folds 0 à 4).
* Comment télécharger : bibliothèque Hugging Face `datasets`, ou chargeur `torchgeo.datasets.CaBuAr`.

### Remarques / limites

* Une seule région (Californie) : fort biais géographique.
* Niveau de traitement et nombre de bandes probablement différents de Sen1Floods11 (L1C, 13 bandes) : réflectances non directement comparables.
* Pas de radar Sentinel-1.
* Risque de confusion entre cicatrices d'anciens feux, sols nus et zones urbaines.

## Décisions

* Source principale : Sen1Floods11, uniquement la partition HandLabeled (446 tuiles), pour la qualité des étiquettes.
* Test « zone jamais vue » : la partition Bolivie, tenue à l'écart de tout entraînement.
* Gestion des masques : `ignore_index = -1` dans la fonction de perte.
* Normalisation : S2 divisé par 10 000 ; S1 laissé en dB puis écrêté et standardisé ; NaN remplacés et masqués. Statistiques calculées sur le jeu d'entraînement uniquement.
* CaBuAr : bonus éventuel, en n'utilisant que l'image post-incendie (tâche mono-date cohérente avec Sen1Floods11).

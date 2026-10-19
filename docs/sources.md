\# Sources de données



\## Sen1Floods11



\* Lien : \[https://github.com/cloudtostreet/Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11)

\* Licence : CC0 1.0 Universal (Domaine public)

\* Article à citer : Bonafilia, D., Tellman, B., Anderson, T., \& Issenberg, E. (2020). \*Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1.\* IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW).

\* Capteurs et bandes : Sentinel-1 SAR (2 bandes : VV, VH en dB) et Sentinel-2 MSI (13 bandes spectrales : B1 à B12, incluant B8A).

\* Niveau de traitement S2 (L1C / L2A) : L1C (réflectance au sommet de l'atmosphère — \*Top-Of-Atmosphere\*).

\* Taille et résolution des tuiles : 512 × 512 pixels à 10 m de résolution spatiale.

\* Valeurs du masque : -1 (No data / nuages), 0 (pas d'eau), 1 (eau / inondation).

\* Nombre d'événements / pays : 11 événements d'inondations historiques répartis sur 11 pays (Bolivie, Cambodge, Ghana, Inde, Pakistan, Paraguay, Somalie, Espagne, Sri Lanka, États-Unis, Viêt Nam).

\* Nombre de tuiles annotées à la main : 446 tuiles réparties en train, val et test officiels (le reste du jeu global repose sur \~4 385 tuiles en étiquetage faible).

\* Comment télécharger : Via le bucket Google Cloud public (`gcloud storage rsync -r gs://sen1floods11/v1.1/data/flood\_events/HandLabeled/ ...`).

\* Remarques / limites : Les nuages non masqués sur le L1C peuvent fausser la détection optique ; la végétation dense masque l'eau en SAR bande C ; nécessité d'utiliser `ignore\_index = -1` lors du calcul des pertes.







\## CaBuAr



\* Lien : \[https://huggingface.co/datasets/DarthReca/california\_burned\_areas](https://huggingface.co/datasets/DarthReca/california\_burned\_areas)

\* Licence : CDLA-Permissive-2.0 (code associé sous licence MIT).

\* Article à citer : Recabarren, D., et al. (2024). \*CaBuAr: California Burned Areas dataset for semantic segmentation of wildland fires.\*

\* Capteurs et bandes : Sentinel-2 MSI, 12 bandes spectrales disponibles en deux dates (pre-fire et post-fire).

\* Niveau de traitement S2 (L1C / L2A) : L2A (réflectance au sol corrigée de l'atmosphère — \*Bottom-Of-Atmosphere\*).

\* Taille et résolution des tuiles : Imagettes de 512 × 512 pixels à 20 m (ou rééchantillonnées selon les configurations).

\* Format des fichiers : Fichiers conteneurs HDF5 (`.h5` / `.hdf5`), accessibles via `datasets` ou `torchgeo`.

\* Valeurs du masque : 0 (non brûlé), 1 (zone brûlée).

\* Zone géographique et période : État de Californie (États-Unis), \~340 événements d'incendies CAL FIRE entre 2015 et 2022.

\* Comment télécharger : Via Python avec la bibliothèque Hugging Face `datasets` ou le chargeur `torchgeo.datasets.CaBuAr`.

\* Remarques / limites : Risque de confusion entre cicatrices de feux anciens et sols nus/zones urbaines ; dépendance obligatoire à `hdf5plugin` et `h5py` pour l'ouverture des fichiers.







\## Décisions



\* Sous-ensemble retenu pour Sen1Floods11 : Utilisation stricte de la partition HandLabeled (446 tuiles) pour garantir la qualité des étiquettes et éviter le bruit des étiquetages faibles.

\* Gestion des masques : Paramétrage de la fonction de perte avec `ignore\_index = -1` pour Sen1Floods11 afin d'exclure les nuages et zones hors-champ.

\* Normalisation : Conversion des réflectances entières en flottants normalisés sur \[0, 1] pour stabiliser l'apprentissage.

\* Stratégie CaBuAr : Utilisation initiale de la configuration post-fire seule pour garder une tâche mono-date cohérente avec Sen1Floods11.


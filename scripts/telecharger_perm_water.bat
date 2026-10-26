@echo off
REM Étape 2.2 : scènes d'eau permanente de Sen1Floods11 (814 tuiles dans le monde, hors crue).
REM A lancer depuis la racine du dépôt, .venv activé.
echo Contenu du dossier distant :
call gsutil ls gs://sen1floods11/v1.1/data/perm_water/
call gsutil du -sh gs://sen1floods11/v1.1/data/perm_water/
echo.
echo Telechargement dans data\sen1floods11\perm_water ...
call gsutil -m cp -r gs://sen1floods11/v1.1/data/perm_water data\sen1floods11\
echo Termine.

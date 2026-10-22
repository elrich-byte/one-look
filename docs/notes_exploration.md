# Notes d'exploration Sen1Floods11

Tuiles observées : les 10 tuiles tirées au hasard dans `notebooks/exploration_01.ipynb` (cellule d'affichage, `random.seed(0)`).

**Lecture des figures.** Dans la case « Masque eau », la palette va de -1 à 1 : **blanc = pas de donnée (-1)**, **bleu moyen = pas d'eau (0)**, **bleu foncé = eau (1)**. Le fond bleu moyen n'est donc pas de l'eau. Dans « Eau permanente », blanc = 0 et bleu foncé = 1.

## Observations par tuile

| Tuile | Catégorie | Crue visible ? | Eau permanente ? | Nuages S2 ? | NaN S1 ? | Piège possible |
|---|---|---|---|---|---|---|
| USA_770353 | crue modérée | oui : chenaux et bras morts en eau autour du fleuve | oui, grand fleuve (méandre) | légère brume | non | le fleuve permanent, bien vu en S1 comme dans le masque |
| Pakistan_474121 | rivière, peu de crue | non | oui, rivière en bas | **oui, très nuageux** | non | masque en grande partie à -1 sous les nuages, alors que S1 voit nettement la rivière |
| USA_181503 | aucune crue | non (quelques points d'eau minuscules) | non | non | non | **parcelles agricoles très sombres en S1**, qui ressemblent à de l'eau |
| Paraguay_1056717 | rivière | rivière en eau | **quasi absente du JRC** | non | non | **rivière étroite absente de l'eau permanente** : la formule la compte comme crue |
| Ghana_124834 | aucune eau | non | non | légère brume | non | radar très granuleux (speckle), aucune eau à détecter |
| India_952728 | peu de crue | petites zones d'eau dispersées | presque pas | **oui, nombreux petits nuages + ombres** | non | grandes zones -1 sous les nuages ; ombres de nuages sombres en S2 |
| Paraguay_683296 | lac + rivière | bord du lac plus étendu que le JRC | oui, lac en haut à droite | un nuage isolé | non | **ombre de nuage noire en S2**, qui ressemble à un plan d'eau |
| Paraguay_44682 | aucune crue | non (petite rivière en haut à gauche) | non | non | non | parcelles sombres en S1 ; sols sombres en S2 en bas de la tuile |
| Pakistan_760290 | aucune eau | non | non | nuages en bas | non | sol aride avec zones sombres en S1 ; masque -1 sous les nuages |
| USA_347609 | petites mares | petites mares dispersées | quelques-unes seulement | non | **oui, bord gauche** (S2 noir aussi) | bord de fauchée : bande sans donnée dans S1, S2 et le masque ; mares absentes du JRC comptées comme crue |

## Comptage (sur 10 tuiles)

- S2 nuageux mais S1 propre : **4** (Pakistan_474121, India_952728 nettement ; Paraguay_683296 et Pakistan_760290 partiellement).
- S1 avec NaN mais S2 propre : **0**. Le seul cas de NaN (USA_347609) touche aussi S2 et le masque, sur la même bande : c'est un bord de fauchée.
- Les deux inutilisables : **1, partiellement** (USA_347609, environ un quart de la tuile).
- Tuiles avec une crue nette : **1 à 2** sur 10 (USA_770353, éventuellement USA_347609). La plupart des tuiles contiennent peu ou pas d'eau de crue.

## Conclusions

1. **Le radar voit bien l'eau étendue, même sous les nuages** (rivière de Pakistan_474121), mais **il ne suffit pas seul** : les champs labourés, les sols nus et les sols arides sont aussi sombres en S1 (USA_181503, Paraguay_44682, Pakistan_760290). L'optique aide à lever le doute quand elle est dégagée. Le speckle rend les petites mares difficiles à voir.
2. **Le piège le plus fréquent** : en S1, les **parcelles agricoles et sols sombres** ; en S2, les **nuages et leurs ombres**. Les annotations ont été faites à partir de S2 : sous les nuages, le masque est à -1, même quand le radar voit l'eau.
3. **Le modèle risque de se tromper** :
   - sur les **rivières étroites et petites mares absentes du JRC** (Paraguay_1056717, USA_347609) : la règle « crue = masque − JRC » les compte comme crue, ce qui peut apprendre de fausses alertes au modèle. Le JRC (résolution 30 m) manque les cours d'eau fins ;
   - sur les **ombres de nuages** (Paraguay_683296) et les **sols sombres** en radar ;
   - sur le **déséquilibre** : très peu de tuiles avec une vraie crue dans ce tirage.

## Pistes pour la suite

- Vérifier ces impressions sur l'ensemble des 446 tuiles avec le tableau `df` (cellule « par catégories »).
- Étape 1.6 : envisager d'ignorer les petites surfaces de « crue » (seuil minimal) pour limiter l'effet des rivières absentes du JRC, ou dilater le masque JRC de quelques pixels.
- Étape 1.7 : ajouter « parcelles agricoles sombres en radar », « ombres de nuages » et « rivières étroites » à la liste des négatifs difficiles.

## Figures

Dans `docs/figures/` : USA_770353 (crue autour d'un fleuve), Pakistan_474121 (nuages, radar efficace), USA_181503 (champs sombres en radar), Paraguay_1056717 (rivière absente du JRC), Paraguay_683296 (ombre de nuage).

"""Étapes 2.1 à 2.4 : rassembler toutes les sources dans data/index.csv.

Une ligne par tuile, avec : source, catégorie, classe (INONDATION / FEU / NORMAL / GRIS /
ECARTEE / INCONNU), ensemble (train / val / test / inconnu) et chemins S1 / S2.

Règles de découpage (aucune fuite géographique) :
- Sen1Floods11 : par pays (config.py).
- Scènes hors crue sur les mêmes lieux : même ensemble que leur pays.
- Scènes mondiales (eau permanente, négatifs difficiles) : par blocs de 10° x 10°,
  70 % train / 15 % val / 15 % test, tirage déterministe.
- Scènes inconnues : ensemble « inconnu », jamais utilisé à l'entraînement.

Les sources non encore téléchargées sont simplement ignorées.
Usage : python src/construire_index.py
"""
import hashlib
import math

import pandas as pd

from config import DATA, DOCS, MAILLE_DEG, PAYS_TEST, PAYS_VAL


def ensemble_pays(pays):
    return "test" if pays in PAYS_TEST else "val" if pays in PAYS_VAL else "train"


def ensemble_maille(lon, lat):
    cle = f"{math.floor(lon / MAILLE_DEG)}_{math.floor(lat / MAILLE_DEG)}"
    u = int(hashlib.md5(cle.encode()).hexdigest(), 16) % 100
    return "train" if u < 70 else "val" if u < 85 else "test"


def sen1floods11():
    df = pd.read_csv(DATA / "index_sen1floods11.csv")
    stats = pd.read_csv(DOCS / "stats_tuiles.csv")[["tile", "perm_pct"]]
    df = df.merge(stats, on="tile")
    cat = "crue_evenement"
    df["categorie"] = cat
    normal = df["classe"] == "NORMAL"
    df.loc[normal, "categorie"] = "normal_pendant_crue"
    df.loc[normal & (df["perm_pct"] > 1), "categorie"] = "normal_eau_permanente"
    df.loc[normal & (df["nuage_pct"] > 30), "categorie"] = "normal_nuageux"
    df = df.rename(columns={"tile": "id", "label": "masque"})
    return df[["id", "source", "categorie", "pays", "ensemble", "classe", "s1", "s2", "masque"]]


def perm_water():
    """814 tuiles d'eau permanente converties par src/convertir_perm_water.py, si présentes.

    Radar seul, calibration approximative : ensemble « eval_eau_permanente »,
    utilisé seulement pour mesurer les fausses alertes, jamais pour l'entraînement.
    """
    dossier = DATA / "sen1floods11" / "perm_water_db"
    if not dossier.exists():
        return pd.DataFrame()
    lignes = []
    for p in sorted(dossier.glob("perm_*_S1.tif")):
        lignes.append(dict(
            id=p.stem.replace("_S1", ""), source="sen1floods11_perm_water",
            categorie="eau_permanente_mondiale", pays="", ensemble="eval_eau_permanente",
            classe="NORMAL", s1=f"sen1floods11/perm_water_db/{p.name}", s2="", masque=""))
    return pd.DataFrame(lignes)


# Vérification visuelle des feux téléchargés (docs/figures/feux_copernicus.png) :
# aucune brûlure visible dans la tuile, ou tuile couverte de nuages -> écartés.
FEUX_NON_CONFIRMES = {"feu_chiquitania_2019", "feu_kabylie_2021", "feu_fort_mcmurray_2016", "feu_manavgat_2021"}


def scenes():
    """Scènes openEO (hors crue, négatifs, inconnues) et CaBuAr, si téléchargées."""
    dossier = DATA / "scenes"
    lignes = []
    if (dossier / "manifest.csv").exists():
        for _, r in pd.read_csv(dossier / "manifest.csv").iterrows():
            s1, s2 = dossier / f"{r['id']}_S1.tif", dossier / f"{r['id']}_S2.tif"
            if not (s1.exists() or s2.exists()):
                continue
            if r["classe"] == "INCONNU":
                ens = "inconnu"
            elif r["groupe"] == "fixe":
                ens = r["ensemble_fixe"]
            elif r["groupe"] == "pays":
                ens = ensemble_pays(r["pays"])
            else:
                ens = ensemble_maille(r["lon"], r["lat"])
            classe = "ECARTEE" if r["id"] in FEUX_NON_CONFIRMES else r["classe"]
            lignes.append(dict(id=r["id"], source="copernicus_openeo", categorie=r["categorie"],
                               pays=r["pays"] if isinstance(r["pays"], str) else "", ensemble=ens,
                               classe=classe,
                               s1=f"scenes/{s1.name}" if s1.exists() else "",
                               s2=f"scenes/{s2.name}" if s2.exists() else "", masque=""))
    if (dossier / "manifest_cabuar.csv").exists():
        # Le pli 0 de CaBuAr va en validation (les feux de validation hors Californie sont
        # trop peu nombreux pour choisir le meilleur modèle) ; jamais en test.
        for _, r in pd.read_csv(dossier / "manifest_cabuar.csv").iterrows():
            ens_cabuar = "val" if int(r.get("pli", -1)) == 0 else "train"
            # CaBuAr (Californie uniquement) sert à l'apprentissage des feux ; les feux
            # hors Californie (Copernicus) servent à la validation et au test.
            lignes.append(dict(id=r["id"], source="cabuar", categorie="zone_brulee", pays="USA-Californie",
                               ensemble=ens_cabuar, classe="FEU", s1="",
                               s2=f"scenes/{r['id']}_S2.tif", masque=f"scenes/{r['id']}_masque.tif"))
            avant = r["id"].replace("cabuar_", "cabuar_avant_")
            if (dossier / f"{avant}_S2.tif").exists():
                lignes.append(dict(id=avant, source="cabuar", categorie="avant_incendie", pays="USA-Californie",
                                   ensemble=ens_cabuar, classe="NORMAL", s1="", s2=f"scenes/{avant}_S2.tif", masque=""))
    return pd.DataFrame(lignes)


def main():
    parties = [sen1floods11(), perm_water(), scenes()]
    df = pd.concat([p for p in parties if len(p)], ignore_index=True).fillna("")
    df.to_csv(DATA / "index.csv", index=False)
    print(f"{len(df)} tuiles -> {DATA / 'index.csv'}\n")
    print(pd.crosstab([df["source"], df["ensemble"]], df["classe"], margins=True).to_string())
    utiles = df[df["classe"].isin(["INONDATION", "FEU", "NORMAL"])]
    print("\nPour l'entraînement et l'évaluation (hors zone grise et écartées) :")
    print(pd.crosstab(utiles["ensemble"], utiles["classe"], margins=True).to_string())


if __name__ == "__main__":
    main()

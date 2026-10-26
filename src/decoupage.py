"""Étape 2.1 : découpage géographique des tuiles Sen1Floods11 par pays.

Lit docs/stats_tuiles.csv, applique la règle d'alerte du cadrage et écrit
data/index_sen1floods11.csv (une ligne par tuile : classe, ensemble, chemins).

Usage : python src/decoupage.py
"""
import pandas as pd

from config import (DOCS, DATA, PAYS_TEST, PAYS_TRAIN, PAYS_VAL, SEUIL_ALERTE_PCT,
                    SEUIL_NORMAL_PCT, VALIDE_MIN_PCT)


def classe(ligne):
    if ligne["valide_pct"] < VALIDE_MIN_PCT:
        return "ECARTEE"
    c = ligne["crue_robuste_pct"]
    if c < SEUIL_NORMAL_PCT:
        return "NORMAL"
    if c >= SEUIL_ALERTE_PCT:
        return "ALERTE"
    return "GRIS"


def ensemble(pays):
    if pays in PAYS_TEST:
        return "test"
    if pays in PAYS_VAL:
        return "val"
    if pays in PAYS_TRAIN:
        return "train"
    raise ValueError(f"Pays non affecté : {pays}")


def main():
    df = pd.read_csv(DOCS / "stats_tuiles.csv")
    assert not set(PAYS_TEST) & set(PAYS_VAL) | set(PAYS_TEST) & set(PAYS_TRAIN) | set(PAYS_VAL) & set(PAYS_TRAIN)

    df["classe"] = df.apply(classe, axis=1)
    df["ensemble"] = df["pays"].map(ensemble)
    df["source"] = "sen1floods11"
    df["categorie"] = "crue_evenement"
    for dossier, suffixe in [("S1Hand", "s1"), ("S2Hand", "s2"), ("LabelHand", "label"), ("JRCWaterHand", "jrc")]:
        df[suffixe] = [f"sen1floods11/{dossier}/{t}_{dossier}.tif" for t in df["tile"]]

    cols = ["tile", "source", "categorie", "pays", "ensemble", "classe", "crue_robuste_pct",
            "valide_pct", "nuage_pct", "s1_nan_pct", "split_officiel", "s1", "s2", "label", "jrc"]
    df[cols].to_csv(DATA / "index_sen1floods11.csv", index=False)

    print(pd.crosstab([df["ensemble"], df["pays"]], df["classe"], margins=True))
    print()
    print(pd.crosstab(df["ensemble"], df["classe"], margins=True))
    fuite = df.groupby("pays")["ensemble"].nunique().max()
    print("\nPays présents dans plusieurs ensembles :", "aucun" if fuite == 1 else "ERREUR")


if __name__ == "__main__":
    main()

"""Étapes 2.2 à 2.4 : télécharger des scènes hors crue, des négatifs difficiles et
des scènes « inconnues » au même format que Sen1Floods11.

Source : Copernicus Data Space Ecosystem via openEO (compte gratuit nécessaire :
https://dataspace.copernicus.eu). Chaque scène = une tuile de 512 x 512 pixels à
~10 m (EPSG:4326, comme Sen1Floods11) :
  data/scenes/<id>_S1.tif : 2 bandes VV, VH en dB (sigma0, NaN = pas de donnée)
  data/scenes/<id>_S2.tif : 13 bandes S2 L1C, réflectance x 10 000
Un manifeste data/scenes/manifest.csv décrit chaque scène.

Usage (depuis la racine du dépôt, .venv activé) :
  python -m pip install openeo
  python src/scenes_openeo.py --liste          # affiche les scènes prévues, sans rien télécharger
  python src/scenes_openeo.py                  # télécharge tout (reprend là où il s'est arrêté)
  python src/scenes_openeo.py --max 3          # essai sur 3 scènes

Dates toutes antérieures à 2022 : à partir du 25/01/2022, les produits S2 L1C
ont un décalage de +1000 sur les valeurs (baseline 04.00) qui les rendrait
incomparables avec Sen1Floods11.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, PAYS_TEST, PAYS_TRAIN, PAYS_VAL  # noqa: E402

RES = 8.983152841195215e-05     # résolution de Sen1Floods11 (degrés)
TAILLE = 512
COTE = RES * TAILLE              # ~0,046 degré
SORTIE = DATA / "scenes"
BANDES_S2 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]

# Fenêtres hors crue (saison sèche ou basses eaux) par événement Sen1Floods11.
SAISON_SECHE = {
    "USA": ("2020-09-01", "2020-10-31"),       # Missouri, basses eaux d'automne
    "India": ("2016-12-01", "2017-04-15"),     # Assam, saison sèche (fenêtre large : brumes d hiver)
    "Paraguay": ("2020-08-01", "2020-10-15"),  # fleuve Paraguay, étiage record 2020
    "Ghana": ("2019-01-05", "2019-02-28"),     # Volta Blanche, harmattan
    "Nigeria": ("2019-01-05", "2019-02-28"),   # Bénoué, saison sèche
    "Pakistan": ("2017-11-01", "2017-12-31"),  # Indus, hors mousson
    "Spain": ("2020-06-01", "2020-07-31"),     # Murcie, été sec
    "Mekong": ("2019-03-01", "2019-04-30"),    # Tonlé Sap, basses eaux
    "Bolivia": ("2018-08-01", "2018-09-30"),   # Béni, saison sèche
    "Sri-Lanka": ("2018-02-01", "2018-03-31"), # entre deux moussons
    "Somalia": ("2019-01-05", "2019-02-28"),   # Shabelle, jilaal (saison sèche)
}
NB_PAR_PAYS = {**{p: 4 for p in PAYS_TRAIN}, **{p: 3 for p in PAYS_VAL + PAYS_TEST}}

# Négatifs difficiles hors Sen1Floods11 : (id, catégorie, lon, lat, début, fin, nuages_max)
NEGATIFS = [
    ("neg_neige_alpes", "neige", 6.95, 45.95, "2021-02-01", "2021-03-15", 20),
    ("neg_neige_rocheuses", "neige", -106.0, 39.6, "2021-01-15", "2021-03-01", 20),
    ("neg_relief_himalaya", "relief", 85.3, 28.2, "2020-10-15", "2020-12-15", 15),
    ("neg_relief_andes", "relief", -72.5, -13.3, "2020-06-01", "2020-08-15", 15),
    ("neg_relief_pyrenees", "relief", 0.6, 42.7, "2020-07-01", "2020-08-31", 15),
    ("neg_desert_sahara", "desert_sable", 5.0, 27.0, "2020-10-01", "2020-11-30", 10),
    ("neg_desert_rubalkhali", "desert_sable", 50.5, 20.5, "2020-11-01", "2020-12-31", 10),
    ("neg_aeroport_cdg", "surface_lisse", 2.55, 49.01, "2020-06-01", "2020-07-31", 15),
    ("neg_aeroport_dubai", "surface_lisse", 55.36, 25.25, "2020-11-01", "2020-12-31", 10),
    ("neg_ville_paris", "urbain", 2.35, 48.86, "2020-06-01", "2020-07-31", 15),
    ("neg_ville_lagos", "urbain", 3.38, 6.52, "2020-12-01", "2021-02-15", 30),
    ("neg_riziere_mekong", "riziere", 105.6, 10.3, "2020-07-01", "2020-08-31", 40),
    ("neg_riziere_niigata", "riziere", 138.95, 37.85, "2020-05-10", "2020-06-20", 30),
    ("neg_champs_beauce", "sol_nu_laboure", 1.6, 48.3, "2020-03-01", "2020-04-15", 20),
    ("neg_champs_iowa", "sol_nu_laboure", -93.6, 42.0, "2020-04-01", "2020-05-15", 20),
    ("neg_lac_leman", "eau_permanente", 6.55, 46.45, "2020-06-01", "2020-07-31", 15),
    ("neg_lac_kariba", "eau_permanente", 27.9, -17.0, "2020-07-01", "2020-08-31", 10),
    ("neg_fleuve_rhone", "eau_permanente", 4.8, 44.0, "2020-07-01", "2020-08-31", 15),
    ("neg_littoral_bretagne", "eau_permanente", -4.5, 48.4, "2020-06-01", "2020-08-31", 20),
    ("neg_zone_humide_camargue", "zone_humide", 4.55, 43.5, "2020-06-01", "2020-07-31", 15),
    ("neg_nuages_congo", "nuages", 23.0, 0.5, "2020-10-01", "2020-10-20", 100),
    ("neg_nuages_borneo", "nuages", 114.0, 1.0, "2020-11-01", "2020-11-20", 100),
]

# Scènes « inconnues » (jamais vues à l'entraînement, à signaler INCONNU).
INCONNUS = [
    ("inc_lave_lapalma", "lave_volcan", -17.88, 28.61, "2021-11-01", "2021-12-10", 30),
    ("inc_incendie_evia", "zone_brulee", 23.3, 38.85, "2021-08-20", "2021-09-30", 15),
    ("inc_incendie_australie", "zone_brulee", 150.3, -33.6, "2020-01-20", "2020-03-01", 20),
    ("inc_banquise_groenland", "glace_mer", -51.0, 69.2, "2021-03-15", "2021-04-30", 20),
    ("inc_glacier_islande", "glacier", -16.7, 64.4, "2021-07-01", "2021-08-31", 30),
    ("inc_mine_atacama", "mine_ciel_ouvert", -68.9, -24.27, "2020-10-01", "2020-11-30", 10),
    ("inc_salar_uyuni", "salar", -67.5, -20.2, "2020-05-01", "2020-11-30", 40),
]


def sites_saison_seche():
    """Pour chaque pays : les tuiles les plus inondées, revisitées en saison sèche."""
    index = pd.read_csv(DATA / "index_sen1floods11.csv")
    centres = pd.read_csv(DATA / "centres_tuiles.csv")
    index = index.merge(centres, on="tile")
    sites = []
    for pays, (debut, fin) in SAISON_SECHE.items():
        choix = (index[(index["pays"] == pays) & (index["classe"] == "ALERTE")]
                 .nlargest(NB_PAR_PAYS[pays], "crue_robuste_pct"))
        for _, r in choix.iterrows():
            sites.append(dict(id=f"sec_{r['tile']}", categorie="hors_crue_meme_lieu", pays=pays,
                              lon=r["lon"], lat=r["lat"], debut=debut, fin=fin,
                              nuages_max=40 if pays == "India" else 20,
                              classe="NORMAL", groupe="pays"))
    return sites


def tous_les_sites():
    sites = sites_saison_seche()
    for liste, classe in [(NEGATIFS, "NORMAL"), (INCONNUS, "INCONNU")]:
        for (i, cat, lon, lat, d, f, n) in liste:
            sites.append(dict(id=i, categorie=cat, pays="", lon=lon, lat=lat, debut=d, fin=f,
                              nuages_max=n, classe=classe, groupe="maille"))
    return sites


def emprise(lon, lat):
    return {"west": lon - COTE / 2, "east": lon + COTE / 2,
            "south": lat - COTE / 2, "north": lat + COTE / 2, "crs": "EPSG:4326"}


def ajuster(chemin_brut, chemin_final, en_db):
    """Recadre / complète à 512 x 512 et convertit S1 en dB."""
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(chemin_brut) as src:
        a = src.read().astype("float32")
        t = src.transform
    if src.nodata is not None:
        a[a == src.nodata] = np.nan
    if en_db:
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(a > 0, 10 * np.log10(a), np.nan)
    sortie = np.full((a.shape[0], TAILLE, TAILLE), np.nan if en_db else 0, dtype="float32")
    h, w = min(TAILLE, a.shape[1]), min(TAILLE, a.shape[2])
    sortie[:, :h, :w] = a[:, :h, :w]
    if not en_db:
        sortie = np.nan_to_num(sortie, nan=0)
    profil = dict(driver="GTiff", height=TAILLE, width=TAILLE, count=sortie.shape[0],
                  dtype="float32", crs="EPSG:4326", compress="deflate",
                  transform=from_origin(t.c, t.f, RES, RES))
    with rasterio.open(chemin_final, "w", **profil) as dst:
        dst.write(sortie)
    Path(chemin_brut).unlink()


TENTATIVES = 3
DELAI_MAX = 600   # secondes par requête ; au-delà, on abandonne et on réessaie


def recuperer(connexion, cube, chemin):
    """Télécharge un cube avec une limite de temps et plusieurs tentatives."""
    import time
    for essai in range(1, TENTATIVES + 1):
        try:
            connexion.download(cube.save_result(format="GTiff"), str(chemin), timeout=DELAI_MAX)
            return
        except Exception as e:
            if Path(chemin).exists():
                Path(chemin).unlink()
            definitive = any(m in str(e) for m in ("No tiff for band", "NoDataAvailable", "FileNotFoundError"))
            if essai == TENTATIVES or definitive:
                raise
            print(f"   tentative {essai} ratée ({str(e)[:80]}), nouvel essai dans 30 s", flush=True)
            time.sleep(30)


def cube_s1(connexion, bbox, debut, fin, lat):
    s1 = connexion.load_collection("SENTINEL1_GRD", spatial_extent=bbox,
                                   temporal_extent=[debut, fin], bands=["VV", "VH"])
    # Au-delà de 60° N, le relief SRTM n'existe pas : on prend le relief Copernicus.
    options = {"elevation_model": "COPERNICUS_30"} if abs(lat) > 59 else {}
    s1 = s1.sar_backscatter(coefficient="sigma0-ellipsoid", **options)
    s1 = s1.reduce_dimension(dimension="t", reducer="first")
    return s1.resample_spatial(resolution=RES, projection=4326, method="near")


def telecharger(connexion, site):
    import datetime as dt
    bbox = emprise(site["lon"], site["lat"])
    s1_final = SORTIE / f"{site['id']}_S1.tif"
    s2_final = SORTIE / f"{site['id']}_S2.tif"

    if not s1_final.exists():
        # Fenêtres successives de 12 jours (un cycle de répétition : une seule acquisition,
        # pas de moyenne qui lisserait le chatoiement). Si une fenêtre échoue (pas de VH,
        # pas d'image), on passe à la suivante, jusqu'à la fin de la période.
        debut = dt.date.fromisoformat(site["debut"])
        limite = dt.date.fromisoformat(site["fin"])
        derniere_erreur = None
        while debut < limite:
            fin = min(debut + dt.timedelta(days=12), limite)
            brut = SORTIE / f"{site['id']}_S1_brut.tif"
            try:
                recuperer(connexion, cube_s1(connexion, bbox, debut.isoformat(), fin.isoformat(),
                                             site["lat"]), brut)
                ajuster(brut, s1_final, en_db=True)
                break
            except Exception as e:
                derniere_erreur = e
                print(f"   S1 {debut} -> {fin} indisponible ({str(e)[:70]}), fenêtre suivante", flush=True)
                debut = fin
        else:
            raise RuntimeError(f"aucune acquisition S1 exploitable : {derniere_erreur}")

    if not s2_final.exists():
        s2 = connexion.load_collection("SENTINEL2_L1C", spatial_extent=bbox,
                                       temporal_extent=[site["debut"], site["fin"]],
                                       bands=BANDES_S2, max_cloud_cover=site["nuages_max"])
        s2 = s2.reduce_dimension(dimension="t", reducer="first")
        s2 = s2.resample_spatial(resolution=RES, projection=4326, method="near")
        brut = SORTIE / f"{site['id']}_S2_brut.tif"
        recuperer(connexion, s2, brut)
        ajuster(brut, s2_final, en_db=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--liste", action="store_true", help="afficher les scènes sans télécharger")
    ap.add_argument("--max", type=int, default=None, help="nombre maximal de scènes")
    ap.add_argument("--paralleles", type=int, default=2, help="scènes téléchargées en même temps")
    args = ap.parse_args()

    sites = tous_les_sites()[: args.max]
    SORTIE.mkdir(parents=True, exist_ok=True)
    with open(SORTIE / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sites[0].keys()))
        w.writeheader()
        w.writerows(sites)

    print(f"{len(sites)} scènes prévues -> {SORTIE / 'manifest.csv'}")
    print(pd.DataFrame(sites).groupby(["classe", "categorie"]).size().to_string())
    if args.liste:
        return

    import openeo
    connexion = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
    from concurrent.futures import ThreadPoolExecutor

    a_faire = [s for s in sites if not ((SORTIE / f"{s['id']}_S1.tif").exists()
                                        and (SORTIE / f"{s['id']}_S2.tif").exists())]
    print(f"{len(sites) - len(a_faire)} déjà présentes, {len(a_faire)} à télécharger "
          f"({args.paralleles} en parallèle).", flush=True)
    erreurs = []

    def traiter(k_site):
        k, site = k_site
        print(f"[{k}/{len(a_faire)}] {site['id']} : début", flush=True)
        try:
            telecharger(connexion, site)
            print(f"[{k}/{len(a_faire)}] {site['id']} : OK", flush=True)
        except Exception as e:  # une scène ratée ne doit pas arrêter les autres
            print(f"[{k}/{len(a_faire)}] {site['id']} : ECHEC : {e}", flush=True)
            erreurs.append((site["id"], str(e)[:200]))

    with ThreadPoolExecutor(max_workers=args.paralleles) as pool:
        list(pool.map(traiter, enumerate(a_faire, 1)))
    print(f"\nTerminé. {len(a_faire) - len(erreurs)} scènes OK, {len(erreurs)} échecs. "
          f"Relancer la même commande pour réessayer les échecs.")
    for i, e in erreurs:
        print(f" - {i} : {e}")


if __name__ == "__main__":
    main()

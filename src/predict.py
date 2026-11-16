"""Analyser une tuile en ligne de commande.

Usage :
  python src/predict.py --s2 tuile_S2.tif [--s1 tuile_S1.tif] [--modele cnn_complet]
  python src/predict.py --s2 tuile_S2.tif --modele cnn_rgb
Fichiers attendus : GeoTIFF 512 x 512 ; S2 = 13 bandes (réflectance x 10 000) ;
S1 = 2 bandes VV, VH en dB.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision import Detecteur  # noqa: E402
from donnees import lire  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2"); ap.add_argument("--s1")
    ap.add_argument("--modele", default="cnn_complet")
    a = ap.parse_args()
    s1 = lire(a.s1) if a.s1 else None
    s2 = lire(a.s2) if a.s2 else None
    print(json.dumps(Detecteur(a.modele).decider(s1, s2), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

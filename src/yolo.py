"""Étape 4 : YOLO (détection par rectangles) sur les images RGB.

YOLO ne classe pas l'image : il dessine des rectangles autour des zones inondées
ou brûlées. La décision pour la tuile est la classe du rectangle le plus sûr,
si sa confiance dépasse un seuil (choisi sur val) ; sinon NORMAL.
Les rectangles d'apprentissage sont fabriqués automatiquement à partir des masques :
eau de crue (masque − eau permanente élargie) pour Sen1Floods11, brûlé pour CaBuAr.

Usage : python src/yolo.py preparer
        python src/yolo.py entrainer
        python src/yolo.py evaluer
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, DOCS, RACINE  # noqa: E402
from entrainer import CACHE, CLASSES, metriques  # noqa: E402
from rgb import composer  # noqa: E402

DOSSIER = DATA / "yolo"
TAILLE = 256
MIN_PIXELS = 8           # à 128 px (40 m/pixel) : environ 1,3 ha
NOMS_YOLO = {0: "inondation", 1: "feu"}


def rectangles(masque, classe):
    from scipy import ndimage
    m = ndimage.binary_dilation(masque, iterations=2)          # regroupe les taches proches
    etiq, n = ndimage.label(m)
    boites = []
    for k, sl in enumerate(ndimage.find_objects(etiq), 1):
        if (masque & (etiq == k)).sum() < MIN_PIXELS:
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        h, w = masque.shape
        boites.append((classe, (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h, (y1 - y0) * (x1 - x0)))
    boites.sort(key=lambda b: -b[5])
    return [b[:5] for b in boites[:15]]


def preparer():
    from PIL import Image
    from scipy import ndimage
    index = pd.read_csv(DATA / "index.csv").fillna("")
    lignes = index[index["ensemble"].isin(["train", "val", "test"]) & index["classe"].isin(CLASSES) & (index["s2"] != "")]
    n = 0
    for _, l in lignes.iterrows():
        f = CACHE / f"{l['id']}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        if "s2" not in d:
            continue
        # étiquettes : seulement si un masque existe, ou si la tuile est NORMAL (aucun rectangle)
        if l["classe"] == "NORMAL":
            boites = []
        elif l["source"] == "sen1floods11" and "masque" in d:
            crue = (d["masque"] == 1) & ~ndimage.binary_dilation(d["jrc"] == 1, iterations=1)
            boites = rectangles(crue, 0)
        elif l["source"] == "cabuar" and "masque" in d:
            boites = rectangles(d["masque"] > 0, 1)
        else:
            continue                          # feux Copernicus sans masque : évaluation par tuile seulement
        for sous in ("images", "labels"):
            (DOSSIER / sous / l["ensemble"]).mkdir(parents=True, exist_ok=True)
        Image.fromarray(composer(d["s2"])).resize((TAILLE, TAILLE), Image.BILINEAR).save(
            DOSSIER / "images" / l["ensemble"] / f"{l['id']}.png")
        (DOSSIER / "labels" / l["ensemble"] / f"{l['id']}.txt").write_text(
            "".join(f"{c} {x:.5f} {y:.5f} {w:.5f} {h:.5f}\n" for c, x, y, w, h in boites))
        n += 1
    (DOSSIER / "donnees.yaml").write_text(
        f"path: {DOSSIER.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: inondation\n  1: feu\n", encoding="utf-8")
    print(f"{n} images préparées -> {DOSSIER}")


def entrainer(epoques=40):
    from ultralytics import YOLO
    modele = YOLO("yolov8n.pt")
    modele.train(data=str(DOSSIER / "donnees.yaml"), imgsz=TAILLE, epochs=epoques, batch=16, device="cpu",
                 workers=0, project=str(RACINE / "modeles" / "yolo"), name="rgb", exist_ok=True,
                 plots=False, verbose=False, seed=0, fliplr=0.5, flipud=0.5, degrees=90, mosaic=0.5)


def decisions(modele, ids, seuil):
    from PIL import Image
    pred = []
    for i in ids:
        d = np.load(CACHE / f"{i}.npz")
        img = Image.fromarray(composer(d["s2"])).resize((TAILLE, TAILLE), Image.BILINEAR)
        r = modele.predict(img, imgsz=TAILLE, conf=0.05, verbose=False)[0]
        conf = r.boxes.conf.numpy() if len(r.boxes) else np.array([])
        cls = r.boxes.cls.numpy().astype(int) if len(r.boxes) else np.array([], int)
        meilleur = [conf[cls == k].max() if (cls == k).any() else 0.0 for k in (0, 1)]
        k = int(np.argmax(meilleur))
        pred.append(k + 1 if meilleur[k] >= seuil else 0)
    return np.array(pred)


def evaluer():
    import time

    import onnxruntime as ort
    from ultralytics import YOLO
    poids = RACINE / "modeles" / "yolo" / "rgb" / "weights" / "best.pt"
    modele = YOLO(str(poids))
    index = pd.read_csv(DATA / "index.csv").fillna("")
    res = dict(modele="yolo_rgb", entree="rgb", parametres=int(sum(p.numel() for p in modele.model.parameters())))
    jeux = {}
    for e in ("val", "test"):
        l = index[(index["ensemble"] == e) & index["classe"].isin(CLASSES) & (index["s2"] != "")]
        l = l[[(CACHE / f"{i}.npz").exists() for i in l["id"]]]
        jeux[e] = (list(l["id"]), np.array([CLASSES.index(c) for c in l["classe"]]))
    meilleur, seuil = -1, 0.25
    for s in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
        f1 = metriques(jeux["val"][1], decisions(modele, jeux["val"][0], s))["f1_macro"]
        if f1 > meilleur:
            meilleur, seuil = f1, s
    res["seuil_confiance"] = seuil
    for e in ("val", "test"):
        p = decisions(modele, jeux[e][0], seuil)
        res[e] = metriques(jeux[e][1], p)
        cat = index.set_index("id").loc[jeux[e][0], "categorie"]
        res[e]["erreurs"] = [dict(id=i, vrai=CLASSES[y], predit=CLASSES[q], categorie=c)
                             for i, y, q, c in zip(jeux[e][0], jeux[e][1], p, cat) if y != q]
    onnx = Path(modele.export(format="onnx", imgsz=TAILLE, simplify=False, verbose=False))
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 1
    s = ort.InferenceSession(str(onnx), opts, providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, TAILLE, TAILLE).astype("float32")
    for _ in range(3):
        s.run(None, {s.get_inputs()[0].name: x})
    t0 = time.perf_counter()
    for _ in range(20):
        s.run(None, {s.get_inputs()[0].name: x})
    res["embarque"] = {"fp32": dict(taille_mo=round(onnx.stat().st_size / 1e6, 2),
                                    ms_par_image_1coeur=round(1000 * (time.perf_counter() - t0) / 20, 1))}
    fichier = DOCS / "resultats_modeles.json"
    tout = json.loads(fichier.read_text(encoding="utf-8")) if fichier.exists() else {}
    tout["yolo_rgb"] = res
    fichier.write_text(json.dumps(tout, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    t = res["test"]
    print(f"yolo_rgb test : F1 {t['f1_macro']:.2f} | fausses alertes {t['fausses_alertes']:.2f} | seuil {seuil}")


if __name__ == "__main__":
    try:  # certificats Windows pour télécharger les poids YOLO
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    {"preparer": preparer, "entrainer": entrainer, "evaluer": evaluer}[sys.argv[1]]()

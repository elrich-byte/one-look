"""Étapes 4 et 5 : évaluer, détecter l'inconnu, alléger (ONNX + INT8), mesurer.

Pour chaque modèle entraîné (modeles/<nom>.pt) :
- métriques sur val et test (pays jamais vus) : précision, rappel, fausses alertes ;
- radar seul (cnn_complet) : fausses alertes sur 814 tuiles d'eau permanente ;
- inconnu : distance de Mahalanobis entre les caractéristiques internes de l'image et
  celles des classes connues (apprises sur train) ; seuil fixé sur val pour garder 95 %
  des images connues ; taux d'images inconnues (scènes + synthétiques) signalées.
  (Le score « énergie », testé d'abord, échoue : AUROC < 0,5, car le bruit aléatoire
  donne des scores très confiants.) ;
- export ONNX puis quantification INT8 ; taille (Mo) et temps par image sur 1 cœur CPU.
Sorties : modeles/<nom>.onnx, modeles/<nom>.int8.onnx, docs/resultats_modeles.json

Usage : python src/evaluer.py cnn_rgb cnn_complet mobilenet_rgb
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, DOCS  # noqa: E402
from donnees import charger_stats  # noqa: E402
from entrainer import (CACHE, CLASSES, CONFIGS, MODELES, Enveloppe, charger_ensemble, charger_modele,  # noqa: E402
                       entree, metriques, predire, predire_avec_caract)
from inconnu import generer  # noqa: E402

TYPES_SYNTHETIQUES = ["bruit_aleatoire", "image_constante", "nuage_total_sans_s1", "bandes_melangees"]


def decider_avec_seuil(logits, seuil):
    p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    alerte = 1 + p[:, 1:].argmax(1)
    return np.where(p[:, 1:].max(1) >= seuil, alerte, 0)


def energie(logits):
    """Score d'inconnu : plus il est haut, plus l'image ressemble aux données connues."""
    m = logits.max(1, keepdims=True)
    return (m + np.log(np.exp(logits - m).sum(1, keepdims=True))).ravel()


class Mahalanobis:
    """Distance aux classes connues dans l'espace des caractéristiques (covariance commune)."""
    def __init__(self, F, Y):
        self.mu = np.stack([F[Y == k].mean(0) for k in range(3)])
        c = sum(np.cov((F[Y == k] - self.mu[k]).T) * (Y == k).sum() for k in range(3)) / len(Y)
        c += np.eye(c.shape[0]) * 1e-3 * np.trace(c) / c.shape[0]          # régularisation
        self.P = np.linalg.inv(c)

    def score(self, F):
        """Plus le score est élevé, plus l'image est loin des données connues."""
        d = [np.einsum("ij,jk,ik->i", F - m, self.P, F - m) for m in self.mu]
        return np.min(d, axis=0)


def auroc(connus, inconnus):
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(connus)), np.zeros(len(inconnus))]
    return float(roc_auc_score(y, np.r_[connus, inconnus]))


def entrees_inconnues(index, mode, stats):
    """Scènes inconnues réelles + entrées synthétiques fabriquées à partir des tuiles de test."""
    X, sources = [], []
    for _, l in index[index["ensemble"] == "inconnu"].iterrows():
        f = CACHE / f"{l['id']}.npz"
        if f.exists():
            x = entree(np.load(f), mode, stats)
            if x is not None:
                X.append(x); sources.append(f"scène : {l['categorie']}")
    test = index[(index["ensemble"] == "test") & (index["source"] == "sen1floods11") & index["classe"].isin(CLASSES)]
    for k, (_, l) in enumerate(test.head(20).iterrows()):
        d = dict(np.load(CACHE / f"{l['id']}.npz"))
        for t in TYPES_SYNTHETIQUES:
            s1, s2 = generer(t, d["s1"], d["s2"], graine=k)
            dd = {k2: v for k2, v in (("s1", s1), ("s2", s2)) if v is not None}
            x = entree(dd, mode, stats)
            if x is not None:
                X.append(x); sources.append(f"synthétique : {t}")
    return np.stack(X), sources


def eau_permanente_radar(modele, stats, seuil_alerte):
    """cnn_complet en radar seul sur l'eau permanente mondiale : part de fausses alertes."""
    import rasterio
    from donnees import normaliser
    index = pd.read_csv(DATA / "index.csv").fillna("")
    perm = index[index["ensemble"] == "eval_eau_permanente"]
    X = []
    for _, l in perm.iterrows():
        with rasterio.open(DATA / l["s1"]) as src:
            s1 = src.read().astype("float32")[:, ::4, ::4]
        x, dispo = normaliser(s1, None, stats)
        if dispo[0]:
            X.append(x)
    p = decider_avec_seuil(predire(modele, np.stack(X)), seuil_alerte)
    return dict(n=len(X), fausses_alertes=float((p != 0).mean()),
                alertes_inondation=float((p == 1).mean()), alertes_feu=float((p == 2).mean()))


def exporter_et_mesurer(nom, modele, Xcalib, Xtest, Ytest):
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    fp32, pre, int8 = MODELES / f"{nom}.onnx", MODELES / f"{nom}.pre.onnx", MODELES / f"{nom}.int8.onnx"
    exemple = torch.from_numpy(Xcalib[:1])
    torch.onnx.export(Enveloppe(modele).eval(), exemple, fp32, input_names=["image"],
                      output_names=["logits", "caracteristiques"], opset_version=17, dynamo=False)
    quant_pre_process(str(fp32), str(pre))

    class Lecteur(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"image": Xcalib[i:i + 1]} for i in range(min(100, len(Xcalib)))])

        def get_next(self):
            return next(self.it, None)

    quantize_static(str(pre), str(int8), Lecteur(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
    pre.unlink()

    res = {}
    for nom_f, f in [("fp32", fp32), ("int8", int8)]:
        opts = ort.SessionOptions(); opts.intra_op_num_threads = 1; opts.inter_op_num_threads = 1
        s = ort.InferenceSession(str(f), opts, providers=["CPUExecutionProvider"])
        for _ in range(5):
            s.run(None, {"image": Xtest[:1]})
        t0 = time.perf_counter()
        for i in range(50):
            s.run(None, {"image": Xtest[i % len(Xtest):i % len(Xtest) + 1]})
        ms = 1000 * (time.perf_counter() - t0) / 50
        logits = np.concatenate([s.run(None, {"image": Xtest[i:i + 1]})[0] for i in range(len(Xtest))])
        res[nom_f] = dict(taille_mo=round(f.stat().st_size / 1e6, 3), ms_par_image_1coeur=round(ms, 2),
                          test=metriques(Ytest, logits.argmax(1)))
    return res


def evaluer(nom):
    cfg = CONFIGS[nom]
    stats = charger_stats()
    index = pd.read_csv(DATA / "index.csv").fillna("")
    modele = charger_modele(nom)
    res = dict(modele=nom, entree=cfg["mode"], parametres=int(sum(p.numel() for p in modele.parameters())))
    jeux = {}
    for ens in ("train", "val", "test"):
        X, Y, ids = charger_ensemble(index, ens, cfg["mode"], stats)
        jeux[ens] = (X, Y, ids)
    logits = {e: predire(modele, jeux[e][0]) for e in jeux}
    # Seuil d'alerte réglé sur val : on n'alerte que si la probabilité de la classe
    # d'alerte la plus probable dépasse le seuil (priorité à la précision).
    seuil_alerte, meilleur = 0.5, -1
    for s in np.arange(0.4, 0.96, 0.05):
        m = metriques(jeux["val"][1], decider_avec_seuil(logits["val"], s))
        score = m["f1_macro"] - 0.5 * m["fausses_alertes"]
        if score > meilleur:
            meilleur, seuil_alerte = score, float(round(s, 2))
    res["seuil_alerte"] = seuil_alerte
    for e in ("val", "test"):
        res[e + "_sans_seuil"] = metriques(jeux[e][1], logits[e].argmax(1))
        res[e] = metriques(jeux[e][1], decider_avec_seuil(logits[e], seuil_alerte))
        # détail des erreurs, pour l'analyse des échecs
        p = decider_avec_seuil(logits[e], seuil_alerte)
        src = index.set_index("id").loc[jeux[e][2]]
        res[e]["erreurs"] = [dict(id=i, vrai=CLASSES[y], predit=CLASSES[q], categorie=c)
                             for i, y, q, c in zip(jeux[e][2], jeux[e][1], p, src["categorie"]) if y != q]

    # Inconnu : énergie (pour mémoire) et Mahalanobis (méthode retenue)
    Xinc, sources = entrees_inconnues(index, cfg["mode"], stats)
    l_inc, f_inc = predire_avec_caract(modele, Xinc)
    _, f_tr = predire_avec_caract(modele, jeux["train"][0])
    _, f_va = predire_avec_caract(modele, jeux["val"][0])
    _, f_te = predire_avec_caract(modele, jeux["test"][0])
    maha = Mahalanobis(f_tr, jeux["train"][1])
    s_va, s_te, s_inc = maha.score(f_va), maha.score(f_te), maha.score(f_inc)
    seuil = float(np.percentile(s_va, 95))
    np.savez(MODELES / f"{nom}.inconnu.npz", mu=maha.mu.astype("float32"), P=maha.P.astype("float32"), seuil=seuil,
             seuil_alerte=res["seuil_alerte"])
    res["inconnu"] = dict(
        methode="Mahalanobis (caractéristiques internes)", seuil=seuil,
        auroc_test=auroc(-s_te, -s_inc),
        auroc_energie_test=auroc(energie(logits["test"]), energie(l_inc)),
        connus_test_gardes=float((s_te <= seuil).mean()),
        inconnus_signales=float((s_inc > seuil).mean()),
        par_source={s: float(np.mean([v > seuil for v, s2 in zip(s_inc, sources) if s2 == s]))
                    for s in sorted(set(sources))})
    if cfg["mode"] == "complet":
        res["eau_permanente_radar_seul"] = eau_permanente_radar(modele, stats, res["seuil_alerte"])
    res["embarque"] = exporter_et_mesurer(nom, modele, jeux["train"][0], jeux["test"][0], jeux["test"][1])
    return res


def main():
    noms = sys.argv[1:] or list(CONFIGS)
    fichier = DOCS / "resultats_modeles.json"
    tout = json.loads(fichier.read_text(encoding="utf-8")) if fichier.exists() else {}
    for nom in noms:
        if not (MODELES / f"{nom}.pt").exists():
            print(f"{nom} : pas encore entraîné"); continue
        r = evaluer(nom)
        tout[nom] = r
        t = r["test"]
        print(f"{nom:14s} test : F1 {t['f1_macro']:.2f} | inondation P {t['INONDATION']['precision']} R {t['INONDATION']['rappel']}"
              f" | feu P {t['FEU']['precision']} R {t['FEU']['rappel']} | fausses alertes {t['fausses_alertes']:.2f}"
              f" | inconnu AUROC {r['inconnu']['auroc_test']:.2f} (énergie {r['inconnu']['auroc_energie_test']:.2f}) | INT8 {r['embarque']['int8']['taille_mo']} Mo,"
              f" {r['embarque']['int8']['ms_par_image_1coeur']} ms", flush=True)
    fichier.write_text(json.dumps(tout, indent=1, ensure_ascii=False, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()

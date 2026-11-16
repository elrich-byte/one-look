"""Décision embarquée : une tuile -> INONDATION / FEU / NORMAL / INCONNU.

Utilise les modèles allégés (modeles/<nom>.int8.onnx) et ONNX Runtime seul :
c'est le code qui tournerait à bord. Étapes :
1. contrôles de format (sinon INCONNU, sans appeler le modèle) ;
2. réduction 512 -> 128 et normalisation (mêmes règles qu'à l'entraînement) ;
3. modèle -> probabilités + caractéristiques internes ; trop loin des classes connues
   (distance de Mahalanobis au-dessus du seuil appris) -> INCONNU.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DOCS, RACINE  # noqa: E402
from donnees import EntreeInvalide, charger_stats, normaliser, verifier  # noqa: E402
from rgb import rgb_float  # noqa: E402

CLASSES = ["NORMAL", "INONDATION", "FEU"]
PAS = 4


class Detecteur:
    def __init__(self, nom="cnn_complet"):
        import onnxruntime as ort
        self.nom = nom
        res = json.loads((DOCS / "resultats_modeles.json").read_text(encoding="utf-8"))[nom]
        self.mode = res["entree"]
        inc = np.load(RACINE / "modeles" / f"{nom}.inconnu.npz")
        self.mu, self.P, self.seuil = inc["mu"], inc["P"], float(inc["seuil"])
        self.seuil_alerte = float(inc["seuil_alerte"])
        opts = ort.SessionOptions(); opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(RACINE / "modeles" / f"{nom}.int8.onnx"), opts,
                                            providers=["CPUExecutionProvider"])
        self.stats = charger_stats()

    def entree(self, s1, s2):
        s1 = None if s1 is None else s1[:, ::PAS, ::PAS]
        s2 = None if s2 is None else s2[:, ::PAS, ::PAS]
        if self.mode == "complet":
            x, dispo = normaliser(s1, s2, self.stats)
            if not dispo.any():
                raise EntreeInvalide("aucun capteur lisible")
            return x
        if s2 is None:
            raise EntreeInvalide("ce modèle a besoin de l'image optique (S2)")
        x = rgb_float(s2)
        if self.mode == "imagenet":
            x = (x - np.array([0.485, 0.456, 0.406])[:, None, None]) / np.array([0.229, 0.224, 0.225])[:, None, None]
        else:
            x = x * 2 - 1
        return x.astype("float32")

    def decider(self, s1, s2):
        try:
            verifier(s1, s2)
            x = self.entree(s1, s2)
        except EntreeInvalide as e:
            return dict(decision="INCONNU", raison=str(e), probabilites=None)
        logits, f = self.session.run(None, {"image": x[None].astype("float32")})
        logits, f = logits[0], f[0]
        p = np.exp(logits - logits.max()); p /= p.sum()
        distance = float(min((f - m) @ self.P @ (f - m) for m in self.mu))
        proba = {c: round(float(v), 3) for c, v in zip(CLASSES, p)}
        if distance > self.seuil:
            return dict(decision="INCONNU", raison=f"image trop différente des données connues "
                        f"(distance {distance:.0f} > seuil {self.seuil:.0f})", probabilites=proba)
        k = 1 + int(p[1:].argmax())
        decision = CLASSES[k] if p[k] >= self.seuil_alerte else "NORMAL"
        return dict(decision=decision, raison="", probabilites=proba)

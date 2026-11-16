# Tableau comparatif (test : pays jamais vus)

Test : 35 inondations (Bolivie, Sri Lanka, Somalie), 4 feux (Grèce, Sardaigne, Australie ×2), 41 tuiles normales. P = précision, R = rappel. Fausses alertes = part des tuiles normales déclarées INONDATION ou FEU.

| Modèle | F1 macro | Inondation P / R | Feu P / R | Fausses alertes | Inconnu (AUROC) | Taille | ms / image (1 cœur) |
|---|---|---|---|---|---|---|---|
| Petit CNN — RGB | 0.31 | 67 % / 17 % | 0 % / 0 % | 12 % | 0.82 | 0.204 Mo (INT8) | 3.29 |
| MobileNetV3 pré-entraîné — RGB | 0.46 | 50 % / 20 % | 40 % / 50 % | 20 % | 0.58 | 1.87 Mo (INT8) | 1.73 |
| YOLOv8n (rectangles) — RGB | 0.54 | 64 % / 46 % | 40 % / 50 % | 29 % | — | 12.61 Mo | 40.2 |
| CLIP sans entraînement — RGB | 0.48 | 61 % / 57 % | 20 % / 75 % | 51 % | — | 605.1 Mo | 148.2 |
| Petit CNN — complet (13 bandes + radar) | 0.61 | 95 % / 60 % | 33 % / 25 % | 5 % | 1.00 | 0.207 Mo (INT8) | 4.84 |

Radar seul (cnn_complet, optique masquée) sur 814 tuiles d'eau permanente dans le monde : 0 % de fausses alertes. Attention : en radar seul, ce modèle n'alerte jamais (rappel 0 sur les inondations de test, voir docs/resultats_capteurs.json) : ce chiffre ne prouve rien.

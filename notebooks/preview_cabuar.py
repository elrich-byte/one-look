import os
from datasets import load_dataset

print("Connexion au jeu de données CaBuAr...")

# On charge le jeu de données en streaming pour ne pas saturer le disque
dataset = load_dataset(
    "DarthReca/california_burned_areas",
    name="post-fire",
    split="train",
    streaming=True,
)

# Aperçu des 3 premiers exemples
print("Lecture de quelques exemples...")
for i, item in enumerate(dataset):
    if i >= 3:
        break
    print(f"\n--- Exemple {i + 1} ---")
    print(f"Clés disponibles : {list(item.keys())}")
    for k, v in item.items():
        if hasattr(v, "shape"):
            print(f"  - {k} : forme {v.shape}, type {v.dtype}")
        elif hasattr(v, "__len__"):
            print(f"  - {k} : longueur {len(v)}")
        else:
            print(f"  - {k} : {v}")

print("\nInspection terminée avec succès !")
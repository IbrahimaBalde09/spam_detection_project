from pathlib import Path
import pandas as pd
import sys

# =========================
# 1. Ajouter le dossier src au path
# =========================
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
DATASET_PATH = BASE_DIR / "data" / "spam.csv"

sys.path.append(str(SRC_DIR))

from preprocess import clean_text

# =========================
# 2. Vérifier que le fichier existe
# =========================
if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Fichier introuvable : {DATASET_PATH}")

print(f"Fichier trouvé : {DATASET_PATH}")

# =========================
# 3. Charger le dataset proprement
# =========================
try:
    df = pd.read_csv(DATASET_PATH)

    # Si une seule colonne, on essaie en TSV
    if df.shape[1] == 1:
        df = pd.read_csv(DATASET_PATH, sep="\t", header=None)

except Exception:
    df = pd.read_csv(DATASET_PATH, sep="\t", header=None)

# =========================
# 4. Harmoniser les colonnes
# =========================
if df.shape[1] == 2:
    df.columns = ["label", "text"]
elif "v1" in df.columns and "v2" in df.columns:
    df = df.rename(columns={"v1": "label", "v2": "text"})
    df = df[["label", "text"]]
elif "label" in df.columns and "text" in df.columns:
    df = df[["label", "text"]]
else:
    raise ValueError(f"Format non reconnu. Colonnes trouvées : {list(df.columns)}")

# =========================
# 5. Nettoyage minimal
# =========================
df = df.dropna(subset=["label", "text"])
df["label"] = df["label"].astype(str).str.strip().str.lower()
df["text"] = df["text"].astype(str)

df = df[df["label"].isin(["ham", "spam"])]

# =========================
# 6. Appliquer le preprocessing
# =========================
df["clean_text"] = df["text"].apply(clean_text)

# =========================
# 7. Afficher des exemples
# =========================
print("\n===== TEXTE ORIGINAL =====")
print(df["text"].iloc[0])

print("\n===== TEXTE NETTOYÉ =====")
print(df["clean_text"].iloc[0])

print("\n===== AUTRE EXEMPLE =====")
print("Original :", df["text"].iloc[1])
print("Nettoyé  :", df["clean_text"].iloc[1])
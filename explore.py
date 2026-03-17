from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. Chemin du fichier
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "spam.csv"

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Fichier introuvable : {DATASET_PATH}")

print(f"Fichier trouvé : {DATASET_PATH}")

# =========================
# 2. Chargement du dataset
# =========================
# On essaie d'abord comme un CSV classique
try:
    df = pd.read_csv(DATASET_PATH)

    # Si le fichier a été lu en une seule colonne, c'est probablement un TSV
    if df.shape[1] == 1:
        df = pd.read_csv(DATASET_PATH, sep="\t", header=None)
except Exception:
    df = pd.read_csv(DATASET_PATH, sep="\t", header=None)

# =========================
# 3. Harmonisation des colonnes
# =========================
if df.shape[1] == 2:
    df.columns = ["label", "text"]
elif "v1" in df.columns and "v2" in df.columns:
    df = df.rename(columns={"v1": "label", "v2": "text"})
    df = df[["label", "text"]]
elif "label" in df.columns and "text" in df.columns:
    df = df[["label", "text"]]
else:
    raise ValueError(
        f"Format non reconnu. Colonnes trouvées : {list(df.columns)}"
    )

# =========================
# 4. Nettoyage minimal
# =========================
df = df.dropna(subset=["label", "text"])
df["label"] = df["label"].astype(str).str.strip().str.lower()
df["text"] = df["text"].astype(str)

# Garder seulement ham/spam si besoin
df = df[df["label"].isin(["ham", "spam"])]

# =========================
# 5. Aperçu général
# =========================
print("\n===== APERÇU DU DATASET =====")
print(df.head())

print("\n===== INFORMATIONS =====")
print(df.info())

print("\n===== VALEURS MANQUANTES =====")
print(df.isnull().sum())

print("\n===== NOMBRE TOTAL DE MESSAGES =====")
print(len(df))

print("\n===== RÉPARTITION DES CLASSES =====")
print(df["label"].value_counts())

# =========================
# 6. Longueur des messages
# =========================
df["length"] = df["text"].apply(len)

print("\n===== STATISTIQUES SUR LA LONGUEUR DES MESSAGES =====")
print(df["length"].describe())

# =========================
# 7. Graphique distribution classes
# =========================
plt.figure(figsize=(6, 4))
df["label"].value_counts().plot(kind="bar")
plt.title("Distribution Spam vs Ham")
plt.xlabel("Classe")
plt.ylabel("Nombre de messages")
plt.tight_layout()
plt.show()

# =========================
# 8. Histogramme longueur messages
# =========================
plt.figure(figsize=(8, 5))
plt.hist(df[df["label"] == "ham"]["length"], bins=50, alpha=0.6, label="ham")
plt.hist(df[df["label"] == "spam"]["length"], bins=50, alpha=0.6, label="spam")
plt.title("Distribution de la longueur des messages")
plt.xlabel("Nombre de caractères")
plt.ylabel("Fréquence")
plt.legend()
plt.tight_layout()
plt.show()
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# =========================
# 1. Chemin du dataset
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "spam.csv"

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Fichier introuvable : {DATASET_PATH}")

print(f"Fichier trouvé : {DATASET_PATH}")

# =========================
# 2. Chargement du dataset
# =========================
try:
    df = pd.read_csv(DATASET_PATH)
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
    raise ValueError(f"Format non reconnu. Colonnes trouvées : {list(df.columns)}")

# =========================
# 4. Nettoyage minimal
# =========================
df = df.dropna(subset=["label", "text"])
df["label"] = df["label"].astype(str).str.strip().str.lower()
df["text"] = df["text"].astype(str)

df = df[df["label"].isin(["ham", "spam"])]

# =========================
# 5. Séparer spam et ham
# =========================
spam_text = " ".join(df[df["label"] == "spam"]["text"])
ham_text = " ".join(df[df["label"] == "ham"]["text"])

# =========================
# 6. Nuage de mots spam
# =========================
wordcloud_spam = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(spam_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_spam, interpolation="bilinear")
plt.axis("off")
plt.title("Nuage de mots - Messages Spam")
plt.tight_layout()
plt.show()

# =========================
# 7. Nuage de mots ham
# =========================
wordcloud_ham = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(ham_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_ham, interpolation="bilinear")
plt.axis("off")
plt.title("Nuage de mots - Messages Non-Spam (Ham)")
plt.tight_layout()
plt.show()
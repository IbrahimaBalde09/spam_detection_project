from pathlib import Path
import pandas as pd
import joblib
import sys

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# 1. Définir les chemins
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATASET_PATH = BASE_DIR / "data" / "spam.csv"
MODELS_DIR = BASE_DIR / "models"

sys.path.append(str(SRC_DIR))

from preprocess import clean_text

# =========================
# 2. Vérifier le dataset
# =========================
if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Fichier introuvable : {DATASET_PATH}")

print(f"Fichier trouvé : {DATASET_PATH}")

# =========================
# 3. Charger le dataset proprement
# =========================
try:
    df = pd.read_csv(DATASET_PATH)

    # Si une seule colonne, on tente en TSV
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

print("\n===== APERÇU =====")
print(df.head())

print("\n===== RÉPARTITION DES CLASSES =====")
print(df["label"].value_counts())

# =========================
# 6. Préprocessing NLP
# =========================
df["clean_text"] = df["text"].apply(clean_text)

print("\n===== EXEMPLE DE TEXTE NETTOYÉ =====")
print(df[["text", "clean_text"]].head(3))

# =========================
# 7. Variables X et y
# =========================
X = df["clean_text"]
y = df["label"]

# =========================
# 8. Séparation train / test
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTaille jeu d'entraînement :", len(X_train))
print("Taille jeu de test :", len(X_test))

# =========================
# 9. Vectorisation TF-IDF
# =========================
vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("\nTF-IDF terminé.")
print("Shape X_train_vec :", X_train_vec.shape)
print("Shape X_test_vec :", X_test_vec.shape)

# =========================
# 10. Entraînement du modèle
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# =========================
# 11. Prédictions
# =========================
y_pred = model.predict(X_test_vec)

# =========================
# 12. Évaluation
# =========================
print("\n===== ACCURACY =====")
print(accuracy_score(y_test, y_pred))

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))

# =========================
# 13. Sauvegarde
# =========================
MODELS_DIR.mkdir(exist_ok=True)

joblib.dump(model, MODELS_DIR / "spam_model.pkl")
joblib.dump(vectorizer, MODELS_DIR / "vectorizer.pkl")

print("\nModèle sauvegardé dans :")
print(MODELS_DIR / "spam_model.pkl")
print(MODELS_DIR / "vectorizer.pkl")
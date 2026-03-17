from pathlib import Path
import sys
import io
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# =========================
# chemins
# =========================
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
DATASET_PATH = BASE_DIR / "data" / "spam.csv"

sys.path.append(str(SRC_DIR))

from preprocess import clean_text

# =========================
# configuration page
# =========================
st.set_page_config(
    page_title="Détection Premium de Spam",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CSS personnalisé
# =========================
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2rem;
            max-width: 1300px;
        }

        .hero-box {
            background: linear-gradient(135deg, #0f172a, #1e293b);
            padding: 32px;
            border-radius: 24px;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
            margin-bottom: 22px;
        }

        .hero-title {
            font-size: 46px;
            font-weight: 800;
            margin-bottom: 10px;
            color: white;
            line-height: 1.1;
        }

        .hero-subtitle {
            font-size: 18px;
            color: #d1d5db;
            line-height: 1.7;
        }

        .section-title {
            font-size: 28px;
            font-weight: 750;
            margin-top: 8px;
            margin-bottom: 12px;
        }

        .custom-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.15);
            margin-bottom: 14px;
        }

        .card-title {
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 10px;
            color: white;
        }

        .muted-text {
            color: #cbd5e1;
            font-size: 15px;
            line-height: 1.7;
        }

        .spam-badge {
            display: inline-block;
            padding: 10px 18px;
            border-radius: 999px;
            font-weight: 800;
            color: white;
            background: linear-gradient(135deg, #ef4444, #dc2626);
            box-shadow: 0 4px 14px rgba(239,68,68,0.35);
            margin-bottom: 12px;
        }

        .ham-badge {
            display: inline-block;
            padding: 10px 18px;
            border-radius: 999px;
            font-weight: 800;
            color: white;
            background: linear-gradient(135deg, #22c55e, #16a34a);
            box-shadow: 0 4px 14px rgba(34,197,94,0.35);
            margin-bottom: 12px;
        }

        .risk-low {
            padding: 14px 16px;
            border-radius: 16px;
            background-color: rgba(34,197,94,0.14);
            border: 1px solid rgba(34,197,94,0.35);
            color: #dcfce7;
            font-weight: 600;
            margin-top: 10px;
            margin-bottom: 10px;
        }

        .risk-medium {
            padding: 14px 16px;
            border-radius: 16px;
            background-color: rgba(250,204,21,0.14);
            border: 1px solid rgba(250,204,21,0.35);
            color: #fef9c3;
            font-weight: 600;
            margin-top: 10px;
            margin-bottom: 10px;
        }

        .risk-high {
            padding: 14px 16px;
            border-radius: 16px;
            background-color: rgba(239,68,68,0.14);
            border: 1px solid rgba(239,68,68,0.35);
            color: #fee2e2;
            font-weight: 600;
            margin-top: 10px;
            margin-bottom: 10px;
        }

        .footer-box {
            margin-top: 36px;
            padding: 18px;
            text-align: center;
            color: #94a3b8;
            font-size: 14px;
            border-top: 1px solid rgba(255,255,255,0.08);
        }

        .mini-note {
            font-size: 13px;
            color: #94a3b8;
            margin-top: 6px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# fonctions utilitaires
# =========================
@st.cache_data
def load_dataset(dataset_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(dataset_path)
        if df.shape[1] == 1:
            df = pd.read_csv(dataset_path, sep="\t", header=None)
    except Exception:
        df = pd.read_csv(dataset_path, sep="\t", header=None)

    if df.shape[1] == 2:
        df.columns = ["label", "text"]
    elif "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "text"})
        df = df[["label", "text"]]
    elif "label" in df.columns and "text" in df.columns:
        df = df[["label", "text"]]
    else:
        raise ValueError(f"Format non reconnu. Colonnes trouvées : {list(df.columns)}")

    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str)
    df = df[df["label"].isin(["ham", "spam"])]

    return df


@st.cache_resource
def load_model():
    model = joblib.load(MODELS_DIR / "spam_model.pkl")
    vectorizer = joblib.load(MODELS_DIR / "vectorizer.pkl")
    return model, vectorizer


@st.cache_data
def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["clean_text"] = prepared["text"].apply(clean_text)
    prepared["length"] = prepared["text"].apply(len)
    return prepared


@st.cache_data
def compute_training_metrics(df: pd.DataFrame):
    local_df = df.copy()

    X = local_df["clean_text"]
    y = local_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer_local = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer_local.fit_transform(X_train)
    X_test_vec = vectorizer_local.transform(X_test)

    logreg = LogisticRegression(max_iter=1000)
    nb = MultinomialNB()

    logreg.fit(X_train_vec, y_train)
    nb.fit(X_train_vec, y_train)

    pred_logreg = logreg.predict(X_test_vec)
    pred_nb = nb.predict(X_test_vec)

    acc_logreg = accuracy_score(y_test, pred_logreg)
    acc_nb = accuracy_score(y_test, pred_nb)

    cm_logreg = confusion_matrix(y_test, pred_logreg, labels=["ham", "spam"])
    report_logreg = classification_report(y_test, pred_logreg, output_dict=True)

    comparison_df = pd.DataFrame(
        {
            "Modèle": ["Logistic Regression", "Naive Bayes"],
            "Accuracy": [acc_logreg, acc_nb]
        }
    )

    return {
        "cm_logreg": cm_logreg,
        "report_logreg": report_logreg,
        "comparison_df": comparison_df
    }


def make_wordcloud(text: str, title: str):
    if not text.strip():
        st.info(f"Aucun texte disponible pour : {title}")
        return

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="white"
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    st.pyplot(fig)


def get_risk_level(prob_spam: float):
    if prob_spam < 0.35:
        return "Faible", "risk-low"
    elif prob_spam < 0.65:
        return "Moyen", "risk-medium"
    else:
        return "Élevé", "risk-high"


def render_gauge(prob_spam: float):
    fig, ax = plt.subplots(figsize=(7, 1.8))
    ax.barh(["Risque spam"], [prob_spam * 100])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Pourcentage")
    ax.set_title("Jauge du risque")
    st.pyplot(fig)


def export_result_text(message, cleaned, prediction, prob_ham, prob_spam):
    output = io.StringIO()
    output.write("RESULTAT ANALYSE SPAM\n")
    output.write("=====================\n\n")
    output.write(f"Message original : {message}\n\n")
    output.write(f"Texte nettoyé : {cleaned}\n\n")
    output.write(f"Prediction : {prediction}\n")
    output.write(f"Probabilité ham : {prob_ham:.4f}\n")
    output.write(f"Probabilité spam : {prob_spam:.4f}\n")
    return output.getvalue()


# =========================
# chargements
# =========================
df = load_dataset(DATASET_PATH)
df = prepare_dataset(df)
model, vectorizer = load_model()
metrics_data = compute_training_metrics(df)

# =========================
# sidebar
# =========================
st.sidebar.markdown("## 📌 Navigation")
page = st.sidebar.radio(
    "Choisir une section",
    [
        "Analyse d’un message",
        "Dashboard du dataset",
        "Performance du modèle",
        "À propos du projet"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Résumé rapide")
st.sidebar.write(f"**Messages** : {len(df)}")
st.sidebar.write(f"**Ham** : {(df['label'] == 'ham').sum()}")
st.sidebar.write(f"**Spam** : {(df['label'] == 'spam').sum()}")
st.sidebar.write("**Modèle principal** : Logistic Regression")
st.sidebar.write("**Vectorisation** : TF-IDF")
st.sidebar.markdown("---")
st.sidebar.caption("Projet NLP • Spam / Non-spam")

# =========================
# page 1 : analyse
# =========================
if page == "Analyse d’un message":
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">📩 Détection Premium de Spam</div>
            <div class="hero-subtitle">
                Application de Machine Learning pour classifier automatiquement les SMS
                en <b>spam</b> ou <b>non-spam</b>, avec probabilités, jauge de risque,
                tableau de bord et export du résultat.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    examples = {
        "Choisir un exemple...": "",
        "Spam 1": "URGENT! You have won a $1000 gift card. Click now to claim your prize.",
        "Spam 2": "Congratulations! You have been selected to receive a FREE iPhone. Reply WIN now.",
        "Spam 3": "You have won a guaranteed cash prize. Call now to collect your reward.",
        "Spam 4": "Your bank account has been suspended. Verify your details immediately.",
        "Ham 1": "Hey, are we still meeting at 7 pm tonight?",
        "Ham 2": "Don't forget to send me the document before tomorrow morning.",
        "Ham 3": "I will call you when I arrive at the station."
    }

    selected_example = st.selectbox("Exemples prêts à tester", list(examples.keys()))
    default_text = examples[selected_example]

    col_left, col_right = st.columns([1.35, 1])

    with col_left:
        st.markdown('<div class="section-title">Message à analyser</div>', unsafe_allow_html=True)
        message = st.text_area(
            "Entrez un message",
            value=default_text,
            height=220,
            placeholder="Exemple : Congratulations! You won a free prize. Call now..."
        )
        analyze = st.button("Analyser le message", use_container_width=True)

    with col_right:
        st.markdown('<div class="section-title">Guide de lecture</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="custom-card">
                <div class="card-title">Ce que l'application affiche</div>
                <div class="muted-text">
                    • la classe prédite<br>
                    • la probabilité de spam<br>
                    • la probabilité de non-spam<br>
                    • une jauge visuelle du risque<br>
                    • le texte nettoyé<br>
                    • un fichier téléchargeable
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if analyze:
        if not message.strip():
            st.warning("Veuillez entrer un message avant de lancer l’analyse.")
        else:
            cleaned = clean_text(message)
            vectorized = vectorizer.transform([cleaned])

            prediction = model.predict(vectorized)[0]
            probabilities = model.predict_proba(vectorized)[0]

            class_order = list(model.classes_)
            prob_ham = float(probabilities[class_order.index("ham")])
            prob_spam = float(probabilities[class_order.index("spam")])

            risk_label, risk_class = get_risk_level(prob_spam)

            st.markdown("---")
            st.markdown('<div class="section-title">Résultats de l’analyse</div>', unsafe_allow_html=True)

            if prediction == "spam":
                st.markdown('<div class="spam-badge">SPAM détecté</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="ham-badge">HAM détecté</div>', unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Prédiction", "SPAM 🚨" if prediction == "spam" else "HAM ✅")
            with m2:
                st.metric("Probabilité spam", f"{prob_spam * 100:.2f}%")
            with m3:
                st.metric("Probabilité ham", f"{prob_ham * 100:.2f}%")
            with m4:
                st.metric("Niveau de risque", risk_label)

            st.markdown(
                f'<div class="{risk_class}"><b>Niveau de risque détecté :</b> {risk_label}</div>',
                unsafe_allow_html=True
            )

            gauge_col, text_col = st.columns([1.2, 1])

            with gauge_col:
                st.markdown("### Jauge de risque")
                render_gauge(prob_spam)

                st.markdown("### Barres de progression")
                st.write("**Spam**")
                st.progress(max(0, min(int(prob_spam * 100), 100)))
                st.write("**Ham**")
                st.progress(max(0, min(int(prob_ham * 100), 100)))

            with text_col:
                st.markdown("### Texte nettoyé")
                st.code(cleaned if cleaned else "(texte vide après nettoyage)")

                st.markdown("### Interprétation")
                if prediction == "spam":
                    st.error(
                        f"Le modèle estime que ce message est probablement un spam avec une confiance de {prob_spam * 100:.2f}%."
                    )
                else:
                    st.success(
                        f"Le modèle estime que ce message est probablement non-spam avec une confiance de {prob_ham * 100:.2f}%."
                    )

            st.markdown("### Dashboard des probabilités")
            prob_df = pd.DataFrame(
                {"Classe": ["Ham", "Spam"], "Probabilité": [prob_ham, prob_spam]}
            )
            st.bar_chart(prob_df.set_index("Classe"))

            export_text = export_result_text(message, cleaned, prediction, prob_ham, prob_spam)
            st.download_button(
                label="Télécharger le résultat (.txt)",
                data=export_text,
                file_name="resultat_analyse_spam.txt",
                mime="text/plain"
            )

            with st.expander("Voir le détail technique"):
                st.write(
                    {
                        "prediction": prediction,
                        "ham_probability": round(prob_ham, 4),
                        "spam_probability": round(prob_spam, 4),
                        "clean_text": cleaned
                    }
                )

# =========================
# page 2 : dataset
# =========================
elif page == "Dashboard du dataset":
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">📊 Dashboard du Dataset</div>
            <div class="hero-subtitle">
                Vue d’ensemble des données utilisées pour entraîner le modèle :
                répartition des classes, statistiques, histogrammes et nuages de mots.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    total_messages = len(df)
    total_ham = int((df["label"] == "ham").sum())
    total_spam = int((df["label"] == "spam").sum())
    spam_ratio = (total_spam / total_messages) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total messages", total_messages)
    c2.metric("Messages ham", total_ham)
    c3.metric("Messages spam", total_spam)
    c4.metric("Taux de spam", f"{spam_ratio:.2f}%")

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.markdown("### Répartition des classes")
        class_counts = df["label"].value_counts()
        st.bar_chart(class_counts)

    with right:
        st.markdown("### Statistiques longueur")
        st.dataframe(df["length"].describe().to_frame(name="Valeur"), use_container_width=True)

    st.markdown("---")

    hist1, hist2 = st.columns(2)

    with hist1:
        st.markdown("### Aperçu des données")
        st.dataframe(df.head(10), use_container_width=True)

    with hist2:
        st.markdown("### Distribution de la longueur des messages")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[df["label"] == "ham"]["length"], bins=50, alpha=0.6, label="ham")
        ax.hist(df[df["label"] == "spam"]["length"], bins=50, alpha=0.6, label="spam")
        ax.set_title("Longueur des messages")
        ax.set_xlabel("Nombre de caractères")
        ax.set_ylabel("Fréquence")
        ax.legend()
        st.pyplot(fig)

    st.markdown("---")

    wc1, wc2 = st.columns(2)

    with wc1:
        st.markdown("### Nuage de mots - Spam")
        spam_text = " ".join(df[df["label"] == "spam"]["text"])
        make_wordcloud(spam_text, "Nuage de mots - Spam")

    with wc2:
        st.markdown("### Nuage de mots - Ham")
        ham_text = " ".join(df[df["label"] == "ham"]["text"])
        make_wordcloud(ham_text, "Nuage de mots - Ham")

# =========================
# page 3 : performance
# =========================
elif page == "Performance du modèle":
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">📈 Performance du Modèle</div>
            <div class="hero-subtitle">
                Évaluation du modèle principal, matrice de confusion, rapport de classification
                et comparaison avec un modèle alternatif.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    comparison_df = metrics_data["comparison_df"]
    cm = metrics_data["cm_logreg"]
    report = metrics_data["report_logreg"]

    perf1, perf2 = st.columns([1, 1])

    with perf1:
        st.markdown("### Comparaison de modèles")
        st.dataframe(comparison_df, use_container_width=True)
        st.bar_chart(comparison_df.set_index("Modèle"))

    with perf2:
        st.markdown("### Matrice de confusion")
        cm_df = pd.DataFrame(
            cm,
            index=["Réel ham", "Réel spam"],
            columns=["Prédit ham", "Prédit spam"]
        )
        st.dataframe(cm_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Prédit ham", "Prédit spam"])
        ax.set_yticklabels(["Réel ham", "Réel spam"])
        ax.set_title("Matrice de confusion")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("### Classification report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

# =========================
# page 4 : à propos
# =========================
elif page == "À propos du projet":
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">🤖 À propos du projet</div>
            <div class="hero-subtitle">
                Présentation du pipeline complet de détection de spam, du prétraitement
                jusqu’au déploiement avec Streamlit.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Objectif")
    st.write(
        "Détecter automatiquement si un SMS est frauduleux (spam) ou légitime (ham) à l’aide d’un pipeline de traitement automatique du langage naturel."
    )

    st.markdown("### Pipeline complet")
    st.markdown(
        """
        1. Chargement du dataset  
        2. Analyse exploratoire  
        3. Nettoyage du texte  
        4. Vectorisation avec TF-IDF  
        5. Entraînement du modèle  
        6. Évaluation  
        7. Déploiement avec Streamlit  
        """
    )

    st.markdown("### Prétraitement appliqué")
    st.markdown(
        """
        - passage en minuscules  
        - suppression des URLs  
        - suppression des chiffres  
        - suppression de la ponctuation  
        - suppression des stopwords  
        - normalisation des espaces  
        """
    )

    st.markdown("### Exemple de nettoyage")
    example_text = "URGENT! You have won 1000 dollars. Call now!!!"
    example_clean = clean_text(example_text)

    ex1, ex2 = st.columns(2)
    with ex1:
        st.write("Texte original")
        st.code(example_text)
    with ex2:
        st.write("Texte nettoyé")
        st.code(example_clean)

    st.markdown("### Modèle principal")
    st.write(
        "Le modèle principal utilisé est une Logistic Regression entraînée sur des vecteurs TF-IDF. Ce choix offre un bon compromis entre simplicité, rapidité et performance."
    )

    st.markdown("### Déploiement")
    st.write(
        "L’application a été développée avec Streamlit pour permettre une utilisation interactive : test d’un message, affichage des probabilités, visualisation du dataset et présentation des performances."
    )

# =========================
# footer
# =========================
st.markdown(
    """
    <div class="footer-box">
        Projet NLP • Détection Spam / Non-spam • Interface Streamlit Premium
    </div>
    """,
    unsafe_allow_html=True
)
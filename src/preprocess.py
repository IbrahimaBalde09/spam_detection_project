import re
import string
import nltk

from nltk.corpus import stopwords

# télécharger les stopwords
nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))


def clean_text(text):
    """
    Nettoie un message texte pour NLP
    """

    # 1. minuscules
    text = text.lower()

    # 2. supprimer URLs
    text = re.sub(r"http\S+", "", text)

    # 3. supprimer chiffres
    text = re.sub(r"\d+", "", text)

    # 4. supprimer ponctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 5. supprimer espaces multiples
    text = re.sub(r"\s+", " ", text).strip()

    # 6. supprimer stopwords
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]

    return " ".join(words)
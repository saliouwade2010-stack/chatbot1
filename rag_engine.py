"""
Moteur RAG : extraction PDF, chunking, retriever TF-IDF, QA extractif.
Toute la logique est mise en cache par Streamlit pour ne charger
le PDF et le modèle QA qu'une seule fois au démarrage.
"""
import re
import pdfplumber
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import nltk

PDF_PATH = "ras01.pdf"
MODEL_NAME = "etalab-ia/camembert-base-squadFR-fquad-piaf"
TAILLE_MOTS = 150
CHEVAUCHEMENT = 30

PATTERN_AGE = re.compile(r"\d+\s*ans\b", re.IGNORECASE)


@st.cache_resource(show_spinner="Chargement et indexation du document...")
def charger_moteur():
    """Extrait le PDF, construit les chunks et la matrice TF-IDF. Mis en cache."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords

    pages_texte = []
    with pdfplumber.open(PDF_PATH) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            texte = page.extract_text()
            if texte:
                pages_texte.append((i, texte))

    chunks = []
    for numero_page, texte in pages_texte:
        mots = texte.split()
        debut = 0
        while debut < len(mots):
            fin = debut + TAILLE_MOTS
            chunk_mots = mots[debut:fin]
            chunks.append({"texte": " ".join(chunk_mots), "page": numero_page})
            debut += TAILLE_MOTS - CHEVAUCHEMENT

    stops_fr = list(stopwords.words("french"))
    textes_chunks = [c["texte"] for c in chunks]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=stops_fr,
        ngram_range=(1, 2),
        max_features=8000,
        sublinear_tf=True,
    )
    matrice_tfidf = vectorizer.fit_transform(textes_chunks)

    return chunks, vectorizer, matrice_tfidf


@st.cache_resource(show_spinner="Chargement du modèle de question-réponse...")
def charger_qa_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    return pipeline("question-answering", model=model, tokenizer=tokenizer)


def rechercher_passages(question, chunks, vectorizer, matrice_tfidf, top_k=5):
    vecteur_question = vectorizer.transform([question])
    similarites = cosine_similarity(vecteur_question, matrice_tfidf)[0]
    indices_tries = similarites.argsort()[::-1][:top_k]
    return [
        {"texte": chunks[i]["texte"], "page": chunks[i]["page"], "score": similarites[i]}
        for i in indices_tries
    ]


def repondre(question, chunks, vectorizer, matrice_tfidf, qa_pipeline,
             top_k=5, poids_retriever=0.4, bonus_pattern=0.25):
    """
    Pipeline complet : recherche des passages candidats, extraction de réponse
    par candidat, sélection du meilleur via un score combiné (retriever + QA
    + bonus de format pour les questions d'âge).
    """
    candidats = rechercher_passages(question, chunks, vectorizer, matrice_tfidf, top_k=top_k)
    meilleure_reponse = None
    meilleur_score_final = -1

    for c in candidats:
        try:
            rep = qa_pipeline(question=question, context=c["texte"])
        except Exception:
            continue

        score_final = poids_retriever * c["score"] + (1 - poids_retriever) * rep["score"]

        if "âge" in question.lower() and PATTERN_AGE.search(rep["answer"]):
            score_final += bonus_pattern

        if score_final > meilleur_score_final:
            meilleur_score_final = score_final
            meilleure_reponse = {
                "answer": rep["answer"].strip(),
                "score_qa": rep["score"],
                "score_retriever": c["score"],
                "score_final": round(score_final, 3),
                "page": c["page"],
                "passage_source": c["texte"],
            }

    return meilleure_reponse

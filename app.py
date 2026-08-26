"""
Chatbot FAQ — Annexe 1 OACI (Licences du personnel)
Interface Streamlit : chat avec historique, indicateur de confiance,
citation de la page source du règlement.
"""
import streamlit as st
from rag_engine import charger_moteur, charger_qa_pipeline, repondre

st.set_page_config(page_title="Chatbot FAQ — Annexe 1 OACI", page_icon="✈️", layout="centered")

st.title("✈️ Chatbot FAQ — Annexe 1 OACI")
st.caption(
    "Posez une question sur les licences du personnel navigant "
    "(Annexe 1 à la Convention relative à l'aviation civile internationale, 14e édition, 2022). "
    "Réponses extraites automatiquement du document — vérifiez toujours la source citée."
)

# Chargement du moteur (mis en cache, ne s'exécute qu'une fois)
chunks, vectorizer, matrice_tfidf = charger_moteur()
qa_pipeline = charger_qa_pipeline()

# Historique de conversation
if "historique" not in st.session_state:
    st.session_state.historique = []

# Affichage de l'historique
for message in st.session_state.historique:
    with st.chat_message(message["role"]):
        st.markdown(message["contenu"])
        if message["role"] == "assistant" and "meta" in message:
            meta = message["meta"]
            if meta is not None:
                niveau = "🟢" if meta["score_qa"] > 0.5 else ("🟡" if meta["score_qa"] > 0.2 else "🔴")
                st.caption(
                    f"{niveau} Confiance : {meta['score_qa']:.0%} — "
                    f"Source : page {meta['page']} de l'Annexe 1"
                )
                with st.expander("Voir le passage source"):
                    st.write(meta["passage_source"])
            else:
                st.caption("🔴 Aucune réponse fiable trouvée dans le document.")

# Zone de saisie
question = st.chat_input("Posez votre question sur les licences du personnel...")

if question:
    st.session_state.historique.append({"role": "user", "contenu": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans le document..."):
            resultat = repondre(question, chunks, vectorizer, matrice_tfidf, qa_pipeline)

        if resultat is None or resultat["score_qa"] < 0.05:
            reponse_texte = (
                "Je ne trouve pas de réponse fiable à cette question dans l'Annexe 1. "
                "Essayez de préciser le type de licence concerné (pilote privé, professionnel, "
                "de ligne, planeur, ballon libre, navigateur, mécanicien navigant...)."
            )
            st.markdown(reponse_texte)
            st.session_state.historique.append({
                "role": "assistant", "contenu": reponse_texte, "meta": None
            })
        else:
            reponse_texte = resultat["answer"]
            st.markdown(reponse_texte)
            niveau = "🟢" if resultat["score_qa"] > 0.5 else ("🟡" if resultat["score_qa"] > 0.2 else "🔴")
            st.caption(
                f"{niveau} Confiance : {resultat['score_qa']:.0%} — "
                f"Source : page {resultat['page']} de l'Annexe 1"
            )
            with st.expander("Voir le passage source"):
                st.write(resultat["passage_source"])

            st.session_state.historique.append({
                "role": "assistant", "contenu": reponse_texte, "meta": resultat
            })

with st.sidebar:
    st.header("À propos")
    st.markdown(
        """
        **Architecture RAG simple**
        1. Extraction du PDF (pdfplumber)
        2. Chunking (150 mots, chevauchement 30)
        3. Recherche par similarité cosinus TF-IDF
        4. Extraction de réponse (CamemBERT fine-tuné SQuAD-FR)
        5. Score combiné retriever + QA + règle de format

        **Document source**
        Annexe 1 — Licences du personnel, OACI, 14e édition (juillet 2022)

        **Limites connues**
        - Confond parfois des clauses lexicalement proches
          mais sémantiquement différentes (ex. durée de validité
          vs âge minimum).
        - Moins fiable sur les questions structurelles/méta
          (nombre de chapitres, éditeur...).
        - Précisez toujours le type de licence dans la question.
        """
    )
    if st.button("Effacer l'historique"):
        st.session_state.historique = []
        st.rerun()

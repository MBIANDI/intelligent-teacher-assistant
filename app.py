import os

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DAT_DIR,
    DB_DIR,
    EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_MODEL_NAME,
    TEMPERATURE,
    USE_OPENAI_EMBEDDINGS,
)
from src.prompt_template import prompt_template
from teacher_assistant.retriever import init_llm, prof_assistant
from teacher_assistant.vectorial_db import vectorial_db_func

load_dotenv()

# ---------------------------------------------------
# LLM, Retriever, Vector DB
# ---------------------------------------------------
llm = init_llm(model_name=OPENAI_MODEL_NAME, temperature=TEMPERATURE)
if USE_OPENAI_EMBEDDINGS:
    db = vectorial_db_func(
        data_path=DAT_DIR,
        model_name=OPENAI_EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        db_path=DB_DIR,
        USE_OPENAI_EMBEDDINGS=USE_OPENAI_EMBEDDINGS,
    )
else:
    db = vectorial_db_func(
        data_path=DAT_DIR,
        model_name=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        db_path=DB_DIR,
    )

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.set_page_config(page_title="Cours NLP - ISSEA", page_icon="🎓", layout="wide")
# st.title("🧠🎓 TutorAI : Posez vos questions sur le cours")


with st.sidebar:
    # --- INFO ÉCOLE & PROF ---
    st.markdown("## 🏫 ISSEA - Cours de NLP")

    # Gestion de la photo de profil
    photo_path = "photo_laura.PNG"  # Assurez-vous que l'image est dans le dossier
    if os.path.exists(photo_path):
        image = Image.open(photo_path)
        st.image(image, width=150, caption="Mme MBIA NDI Marie Thérèse")
    else:
        st.warning("⚠️ Image 'prof_photo.jpg' non trouvée.")
        st.markdown("**Professeur :** Mme MBIA NDI Marie Thérèse")

    st.markdown("---")
    st.markdown(
        """
    **Objectif du bot :** Répondre aux questions des étudiants sur le support de cours officiel.
    """
    )

    st.markdown("---")

# En-tête personnalisé avec HTML/CSS pour un rendu "pro"
st.markdown(
    """
    <div style='background-color:#002b36;padding:20px;border-radius:10px;margin-bottom:20px'>
        <h1 style='color:white;text-align:center;'>🤖🧠🎓 TutorAI - NLP</h1>
        <h3 style='color:#aeb6bf;text-align:center;'>ISSEA - Année 2025/2026</h3>
    </div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# SAISIE & ENVOI
# ---------------------------------------------------
assistant = prof_assistant(
    llm=llm,
    prompt=prompt_template,
    vector_db=db,
)
history = []
# Initialiser l'historique des messages dans l'interface Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Bonjour ! Je suis votre assistant de cours. Posez-moi une question sur le contenu de vos PDFs.",
        }
    ]
# Afficher les messages précédents (pour garder l'historique visuel)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie utilisateur
if query := st.chat_input("Votre question..."):
    # 1. Afficher la question de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 2. Générer la réponse via le RAG
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Je consulte mes notes de cours..."):
            try:
                response = assistant({"question": query}, {"chat_history": history})
                full_response = response["answer"]

                message_placeholder.markdown(full_response)

            except Exception as e:
                full_response = f"Une erreur est survenue : {e}"
                message_placeholder.error(full_response)

    # 3. Sauvegarder la réponse de l'assistant dans l'historique
    st.session_state.messages.append({"role": "assistant", "content": full_response})

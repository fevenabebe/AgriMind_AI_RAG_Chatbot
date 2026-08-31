import os
import tempfile

import streamlit as st

from src.classifier import (
    classify_crop,
    normalize_disease_name,
    MAIZE_CLASSES,
    WHEAT_CLASSES
)

from src.retrieval import retrieve
from src.context import build_context
from src.generator import generate_rag_answer


st.set_page_config(
    page_title="AgriMind AI RAG Chatbot",
    page_icon="🌱",
    layout="wide"
)


st.title("🌱 AgriMind AI RAG Chatbot")

st.markdown(
    """
    **AI-powered crop disease detection and evidence-grounded
    agricultural assistance.**

    Upload a wheat or maize leaf image to obtain a preliminary
    disease prediction, then ask questions using the
    retrieval-augmented chatbot.
    """
)


with st.sidebar:

    st.header("Crop Analysis")

    crop = st.selectbox(
        "Select crop",
        ["wheat", "maize"]
    )

    worker_name = st.text_input(
        "Extension Worker Name",
        value="Extension Worker"
    )

    uploaded_file = st.file_uploader(
        "Upload crop image",
        type=["jpg", "jpeg", "png"]
    )


if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "image_path" not in st.session_state:
    st.session_state.image_path = None

if "current_results" not in st.session_state:
    st.session_state.current_results = []


if uploaded_file is not None:

    st.subheader("Uploaded Image")

    st.image(
        uploaded_file,
        width=400
    )

    if st.button(
        "🔍 Analyze Image",
        type="primary"
    ):

        suffix = os.path.splitext(
            uploaded_file.name
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:

            tmp.write(
                uploaded_file.getbuffer()
            )

            image_path = tmp.name

        st.session_state.image_path = image_path

        try:

            result = classify_crop(
                image_path,
                crop
            )

            st.session_state.prediction = result
            st.session_state.chat_history = []
            st.session_state.current_results = []

        except Exception as e:

            st.error(
                f"Classification failed: {e}"
            )


prediction = st.session_state.prediction


if prediction is not None:

    detected_crop = prediction["crop"]
    disease = prediction["disease"]
    confidence = prediction["confidence"]
    probabilities = prediction["probabilities"]

    disease_display = normalize_disease_name(
        disease
    )

    st.divider()

    st.subheader("🌿 Classification Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Crop",
            detected_crop.title()
        )

    with col2:
        st.metric(
            "Suspected Disease",
            disease_display
        )

    with col3:
        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )


    st.subheader("Classifier Probabilities")

    if detected_crop == "maize":
        class_names = MAIZE_CLASSES
    else:
        class_names = WHEAT_CLASSES


    for class_name, probability in zip(
        class_names,
        probabilities
    ):

        st.write(
            f"**{normalize_disease_name(class_name)}**"
        )

        st.progress(
            float(probability)
        )

        st.caption(
            f"{float(probability):.2%}"
        )


    st.warning(
        """
        The classifier result is a preliminary prediction
        and is not a laboratory diagnosis.

        Please confirm suspected disease through qualified
        extension/pathology services when required.
        """
    )


    st.divider()

    st.subheader("💬 AgriMind Agricultural Assistant")

    st.write(
        f"Hello, {worker_name or 'Extension Worker'}. "
        f"I can help you with questions about "
        f"**{disease_display}** in **{detected_crop}**."
    )


    for message in st.session_state.chat_history:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    question = st.chat_input(
        f"Ask about {disease_display}..."
    )


    if question:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)


        rag_query = f"""
Crop: {detected_crop}
Disease: {disease_display}

Question:
{question}
"""

        try:
            results = retrieve(
                rag_query,
                top_k=8
            )
            st.session_state.current_results = results
        except Exception as e:
            st.error(
                f"Evidence retrieval failed: {e}"
            )
            results = []
            st.session_state.current_results = []


        if not results:

            answer = (
                "I could not find sufficient agricultural "
                "evidence to answer that question. "
                "Please try a more specific question."
            )

        else:

            try:

                context = build_context(
                    results
                )

            except Exception as e:

                st.error(
                    f"Could not prepare retrieved evidence: {e}"
                )

                context = ""


            if context:

                try:

                    answer = generate_rag_answer(
                        rag_query,
                        context
                    )

                except Exception as e:

                    answer = (
                        "The AI response service is "
                        "temporarily unavailable. "
                        "However, relevant agricultural "
                        "evidence was successfully retrieved."
                    )

                    st.error(
                        f"Generation error: {e}"
                    )

            else:

                answer = (
                    "I could not prepare sufficient evidence "
                    "for an answer."
                )


        with st.chat_message("assistant"):

            st.markdown(answer)


        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )
        
        st.rerun()


    if st.session_state.current_results:

        with st.expander(
            "📚 Retrieved Agricultural Evidence"
        ):

            for i, result in enumerate(
                st.session_state.current_results,
                start=1
            ):

                metadata = result.get(
                    "document",
                    {}
                ).get(
                    "metadata",
                    {}
                )

                st.markdown(
                    f"### Evidence {i}"
                )

                st.write(
                    f"**Retrieval score:** "
                    f"{float(result.get('score', 0)):.4f}"
                )

                st.write(
                    f"**Crop:** "
                    f"{metadata.get('crop', 'Unknown')}"
                )

                st.write(
                    f"**Disease:** "
                    f"{metadata.get('disease', 'Unknown')}"
                )

                st.write(
                    f"**Topic:** "
                    f"{metadata.get('topic', 'Unknown')}"
                )

                st.write(
                    f"**Source:** "
                    f"{metadata.get('source_title', 'Unknown')}"
                )

                st.write(
                    f"**Organization/Author:** "
                    f"{metadata.get('source_organization', 'Unknown')}"
                )

                st.write(
                    f"**Publication year:** "
                    f"{metadata.get('publication_year', 'Unknown')}"
                )

                st.write(
                    f"**Source type:** "
                    f"{metadata.get('source_type', 'Unknown')}"
                )

                st.write(
                    f"**Region:** "
                    f"{metadata.get('region', 'Unknown')}"
                )

                st.write(
                    f"**Evidence type:** "
                    f"{metadata.get('evidence_type', 'Unknown')}"
                )

                st.write(
                    f"**Confidence:** "
                    f"{metadata.get('confidence', 'Unknown')}"
                )

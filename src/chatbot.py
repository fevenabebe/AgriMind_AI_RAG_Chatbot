
# ============================================================
# AGRIMIND CHATBOT
# ============================================================

from src.classifier import (
    classify_crop,
    normalize_disease_name,
    MAIZE_CLASSES,
    WHEAT_CLASSES
)

from src.retrieval import retrieve

from src.context import build_context

from src.generator import generate_rag_answer


# ============================================================
# AGRIMIND CHATBOT
# ============================================================

def run_agrimind_chatbot(image_path, crop):

    # --------------------------------------------------------
    # 1. CLASSIFY IMAGE
    # --------------------------------------------------------

    result = classify_crop(
        image_path,
        crop
    )

    detected_crop = result["crop"]
    disease = result["disease"]
    confidence = result["confidence"]
    probabilities = result["probabilities"]

    disease_display = normalize_disease_name(
        disease
    )


    # --------------------------------------------------------
    # 2. DISPLAY HEADER
    # --------------------------------------------------------

    print("\n🌱 AgriMind Chatbot")
    print("=" * 60)

    print(
        f"Crop: {detected_crop}"
    )

    print(
        f"Suspected disease: {disease_display}"
    )

    print(
        f"Confidence: {confidence:.2%}"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # 3. DISPLAY CLASS PROBABILITIES
    # --------------------------------------------------------

    print("\nClassifier probabilities:")

    if detected_crop == "maize":

        class_names = MAIZE_CLASSES

    else:

        class_names = WHEAT_CLASSES


    for class_name, probability in zip(
        class_names,
        probabilities
    ):

        print(
            f"  {class_name}: "
            f"{float(probability):.2%}"
        )


    # --------------------------------------------------------
    # 4. DIAGNOSTIC DISCLAIMER
    # --------------------------------------------------------

    print("\nAgriMind:")

    print(
        "The classifier result is a preliminary "
        "prediction and is not a laboratory diagnosis."
    )

    print(
        "Please confirm suspected disease through "
        "qualified extension/pathology services "
        "when required."
    )


    # --------------------------------------------------------
    # 5. EXTENSION WORKER
    # --------------------------------------------------------

    print()

    worker_name = input(
        "Extension Worker Name: "
    ).strip()


    if not worker_name:

        worker_name = "Extension Worker"


    print(
        f"\nAgriMind: Hello, {worker_name}."
    )

    print(
        f"I can help you with questions about "
        f"{disease_display} in {detected_crop}."
    )

    print(
        "\nAsk a question, or type 'exit' to end the session."
    )


    # --------------------------------------------------------
    # 6. QUESTION LOOP
    # --------------------------------------------------------

    while True:

        print()

        question = input(
            "Extension Worker: "
        ).strip()


        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        if question.lower() in [
            "exit",
            "quit",
            "q",
            "bye"
        ]:

            print(
                "\nAgriMind: Goodbye, "
                f"{worker_name}."
            )

            break


        # ----------------------------------------------------
        # EMPTY QUESTION
        # ----------------------------------------------------

        if not question:

            print(
                "\nAgriMind: Please enter a question "
                "about the suspected disease."
            )

            continue


        # ----------------------------------------------------
        # 7. CREATE RAG QUERY
        # ----------------------------------------------------

        rag_query = f"""
Crop: {detected_crop}
Disease: {disease_display}

Question:
{question}
"""


        # ----------------------------------------------------
        # 8. RETRIEVE EVIDENCE
        # ----------------------------------------------------

        try:

            results = retrieve(
                rag_query,
                top_k=8
            )

        except Exception as e:

            print(
                "\nAgriMind: I encountered an error "
                "while retrieving evidence."
            )

            print(
                f"Error: {e}"
            )

            continue


        # ----------------------------------------------------
        # 9. CHECK RETRIEVAL
        # ----------------------------------------------------

        if not results:

            print(
                "\nAgriMind:"
            )

            print(
                "I could not find sufficient evidence "
                "to answer that question."
            )

            print(
                "Please try a more specific question."
            )

            continue


        # ----------------------------------------------------
        # 10. BUILD CONTEXT
        # ----------------------------------------------------

        try:

            context = build_context(
                results
            )

        except Exception as e:

            print(
                "\nAgriMind: I could not prepare "
                "the retrieved evidence."
            )

            print(
                f"Error: {e}"
            )

            continue


        # ----------------------------------------------------
        # 11. GENERATE RAG ANSWER
        # ----------------------------------------------------

        try:

            answer = generate_rag_answer(
                rag_query,
                context
            )

        except Exception as e:

            print(
                "\nAgriMind: I could not generate "
                "an answer at this time."
            )

            print(
                f"Error: {e}"
            )

            continue


        # ----------------------------------------------------
        # 12. DISPLAY ANSWER
        # ----------------------------------------------------

        print("\nAgriMind:")
        print(answer)

        print(
            "\n" + "-" * 60
        )


# ============================================================
# AGRIMIND AI RAG CHATBOT
# ============================================================

from src.chatbot import run_agrimind_chatbot


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("🌱 AGRIMIND AI RAG CHATBOT")
    print("=" * 60)

    print()
    print("Supported crops:")
    print("  1. Wheat")
    print("  2. Maize")
    print()

    crop_choice = input(
        "Enter crop (wheat/maize): "
    ).strip().lower()

    if crop_choice not in [
        "wheat",
        "maize"
    ]:

        raise ValueError(
            "Unsupported crop. Please enter 'wheat' or 'maize'."
        )

    print()

    image_path = input(
        "Enter image path: "
    ).strip()

    if not image_path:

        raise ValueError(
            "An image path is required."
        )

    run_agrimind_chatbot(
        image_path,
        crop_choice
    )

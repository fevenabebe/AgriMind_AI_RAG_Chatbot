
import os

from google import genai


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )


client = genai.Client(
    api_key=GEMINI_API_KEY
)


MODEL_NAME = "gemini-3.6-flash"


# ============================================================
# SYSTEM INSTRUCTION
# ============================================================

SYSTEM_INSTRUCTION = """
You are AgriMind, an agricultural disease advisory assistant.

Your role is to help agricultural extension workers understand
crop disease information using retrieved evidence.

IMPORTANT RULES:

1. Use the retrieved evidence as the primary source of information.

2. Do not invent facts that are not supported by the retrieved
   evidence.

3. Clearly distinguish Ethiopia-specific evidence from global
   technical references.

4. Do not present an image classifier prediction as a confirmed
   laboratory diagnosis.

5. Do not provide specific pesticide products, rates, doses,
   application schedules, or other chemical recommendations
   unless they are explicitly supported by current Ethiopian
   official guidance in the retrieved evidence.

6. Do not treat foreign recommendations as Ethiopian decision rules.

7. If the retrieved evidence is insufficient to answer the
   question, clearly say that the available evidence is
   insufficient.

8. Give practical, concise answers suitable for an agricultural
   extension worker.

9. When appropriate, recommend confirmation through EIAR,
   the Ministry of Agriculture, regional agricultural bureaus,
   qualified extension workers, or plant pathology specialists.

10. Do not fabricate sources or citations.
"""


# ============================================================
# RAG ANSWER GENERATION
# ============================================================

def generate_rag_answer(
    query,
    context,
    max_retries=3
):

    prompt = f"""
{SYSTEM_INSTRUCTION}

============================================================
RETRIEVED AGRICULTURAL EVIDENCE
============================================================

{context}

============================================================
EXTENSION WORKER QUESTION
============================================================

{query}

============================================================

Answer the extension worker's question using only the
retrieved agricultural evidence.

Keep the answer clear and practical.

If the evidence does not contain enough information to answer
the question, explicitly state that the available evidence is
insufficient rather than guessing.
"""

    last_error = None

    for attempt in range(max_retries):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            if response.text:
                return response.text.strip()

            return (
                "The AI response did not contain any text. "
                "Please try again."
            )

        except Exception as e:

            last_error = e

            error_text = str(e)

            if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:

                return (
                    "The AI response service is temporarily "
                    "unavailable because the Gemini API quota "
                    "has been reached.\n\n"
                    "However, the relevant agricultural evidence "
                    "was successfully retrieved. Please try again "
                    "after the API quota resets."
                )

            if attempt == max_retries - 1:

                return (
                    "I could not generate an AI answer at this "
                    "time. Please try again later."
                )

    return (
        "I could not generate an AI answer at this time."
    )

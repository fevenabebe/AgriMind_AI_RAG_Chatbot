
# ============================================================
# AGRIMIND RAG CONTEXT BUILDER
# ============================================================


def build_context(results):
    """
    Convert retrieved FAISS results into a clean evidence
    context for the RAG generation step.
    """

    context_parts = []

    for i, result in enumerate(results, start=1):

        document = result["document"]
        score = result["score"]

        text = document["text"]

        context_parts.append(
            f"""
Evidence {i}
------------------------------------------------------------
Retrieval Score: {score:.4f}

{text}
"""
        )

    return "\n".join(context_parts)

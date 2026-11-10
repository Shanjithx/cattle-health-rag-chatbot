import os

from dotenv import load_dotenv
from openai import OpenAI

from retrieval import hybrid_search
from guardrails import check_guardrails


# ==========================================
# CONFIGURATION
# ==========================================

load_dotenv()

API_KEY = os.getenv("JARVIS_API_KEY")
BASE_URL = os.getenv("JARVIS_BASE_URL")
MODEL = os.getenv("JARVIS_MODEL", "glm-5.3-flash")

if not API_KEY:
    raise ValueError("JARVIS_API_KEY is missing from .env")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=60
)

# Keep False for clean output. Set True only when debugging retrieval.
SHOW_DIAGNOSTICS = False

# Retrieve a small, reranked evidence set to reduce prompt size and latency.
RETRIEVAL_TOP_K = 5
MAX_CONTEXT_CHARS = 14000
MAX_OUTPUT_TOKENS = 1800


# ==========================================
# RETRIEVAL AND CONTEXT
# ==========================================

def retrieve_context(question, top_k=RETRIEVAL_TOP_K):
    results = hybrid_search(
        question,
        top_k=top_k,
        verbose=SHOW_DIAGNOSTICS
    )
    return results


def build_context(results):
    """Format only the highest-ranked retrieved passages."""
    context_parts = []
    total_chars = 0

    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        passage = result.get("text") or result.get("content") or ""
        if not passage.strip():
            continue

        filename = metadata.get(
            "filename",
            metadata.get("source", "Unknown document")
        )
        page = metadata.get("page", "Unknown")
        category = metadata.get("category", "General")

        block = (
            f"[SOURCE {index}]\n"
            f"Document: {filename}\n"
            f"Category: {category}\n"
            f"Page: {page}\n"
            f"Content: {passage.strip()}\n"
        )

        remaining = MAX_CONTEXT_CHARS - total_chars
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = block[:remaining]

        context_parts.append(block)
        total_chars += len(block)

    return "\n".join(context_parts)


# ==========================================
# SINGLE-PASS ANSWER GENERATION
# ==========================================

def generate_final_answer(question, context):
    """Assess evidence and produce the final answer in one model call.

    This replaces the separate evidence-assessment, fallback-generation,
    fallback-review, and answer-verification calls in the previous pipeline.
    """

    has_context = bool(context.strip())

    prompt = f"""
You are a careful cattle-health and dairy-farming assistant.
Answer the user's question accurately, clearly, and safely.

USER QUESTION:
{question}

LOCAL KNOWLEDGE-BASE EXCERPTS:
{context if has_context else 'No relevant passages were retrieved from the local knowledge base.'}

INSTRUCTIONS:

1. First assess internally whether the excerpts directly and sufficiently
   support an answer to the user's actual question. Do not print this assessment.

2. If the excerpts sufficiently support the answer:
   - Answer using the excerpts.
   - Cite factual information as (Document: filename, page: number).
   - Do not add facts from pretrained knowledge as though they came from the PDFs.

3. If the excerpts support only part of the answer:
   - Give the supported information first, with citations.
   - If you add well-established general cattle/veterinary knowledge from your
     pretrained knowledge, clearly label that portion as general information
     not found in the local documents.
   - Do not guess or fill gaps with uncertain claims.

4. If the excerpts do not support an answer:
   - Use pretrained knowledge only for well-established, general educational
     information that you can answer reliably.
   - Clearly state that the answer is not supported by the local PDFs.
   - If uncertain, say that reliable information is insufficient and recommend
     consulting a qualified veterinarian.

5. Veterinary safety:
   - Do not diagnose an individual animal remotely.
   - Do not prescribe medicines or provide exact drug dosages.
   - Do not invent withdrawal periods, treatment protocols, or clinical findings.
   - For diagnosis, treatment, severe symptoms, or emergencies, recommend a
     qualified veterinarian.
   - Distinguish correlation from causation.
   - Be careful about species: do not treat sheep, goats, pigs, or other animal
     evidence as cattle-specific evidence unless the source explicitly supports it.

6. Answer style:
   - Start with the direct answer.
   - Use concise paragraphs or bullets when useful.
   - Explain relevant details without unnecessary repetition.
   - Cite every factual claim derived from the PDFs.
   - Do not show retrieval scores, confidence labels, internal reasoning,
     processing times, token usage, or verification reports.
   - Return only the final answer.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a cautious cattle-health information assistant. "
                    "Prioritise factual accuracy, species-specific evidence, "
                    "and veterinary safety. Never fabricate citations."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=MAX_OUTPUT_TOKENS,
        extra_body={
            "reasoning_effort": "low",
            "clear_thinking": True
        }
    )

    answer = response.choices[0].message.content or ""
    return answer.strip()


# ==========================================
# MAIN QUESTION PROCESSING
# ==========================================

def ask_question(question):
    question = (question or "").strip()

    if not question:
        print("Please enter a question.")
        return

    if len(question) > 2000:
        print("Please keep your question under 2000 characters.")
        return

    # Keep the domain guardrail, but do not display its internal details.
    allowed, rejection_message = check_guardrails(question)
    if not allowed:
        print(rejection_message)
        return

    if question.lower() in {
        "hello", "hi", "hey", "good morning",
        "good afternoon", "good evening"
    }:
        print("Hello! I can help with cattle health, diseases, nutrition, and dairy farming.")
        return

    try:
        results = retrieve_context(question)
        context = build_context(results) if results else ""
        answer = generate_final_answer(question, context)

        if answer:
            print("\n" + answer)
        else:
            print("I could not generate a reliable answer. Please try rephrasing your question or consult a qualified veterinarian.")

    except Exception as error:
        # Keep the user-facing output concise; detailed exceptions can be
        # inspected during development by temporarily enabling diagnostics.
        if SHOW_DIAGNOSTICS:
            raise
        print(f"Unable to process the question right now: {error}")


# ==========================================
# COMMAND-LINE INTERFACE
# ==========================================

def main():
    print("CATTLE HEALTH ASSISTANT")
    print("Ask questions about cattle diseases, health, nutrition, and dairy farming.")
    print("Type 'exit' to quit.")

    while True:
        try:
            question = input("\nYou: ").strip()

            if question.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            ask_question(question)

        except KeyboardInterrupt:
            print("\nAssistant stopped.")
            break


if __name__ == "__main__":
    main()

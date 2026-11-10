import os
import re
import time

SHOW_DIAGNOSTICS = True
MAX_ANSWER_CONTINUATIONS = 3

from dotenv import load_dotenv
from openai import OpenAI

from retrieval import hybrid_search
from guardrails import check_guardrails


# ==========================================
# 1. CONFIGURATION
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

# Pricing per 1 million tokens (INR)
INPUT_PRICE = 7.11
OUTPUT_PRICE = 23.69


# ==========================================
# 2. HYBRID RETRIEVAL WITH RERANKING
# ==========================================

def retrieve_context(question, top_k=5):

    if SHOW_DIAGNOSTICS:
        print("\nSearching the cattle knowledge base...", flush=True)

    results = hybrid_search(
        question,
        top_k=top_k,
        verbose=SHOW_DIAGNOSTICS
    )

    if SHOW_DIAGNOSTICS:
        print(
            f"\nRetrieved {len(results)} reranked passages.",
            flush=True
        )

    return results


# ==========================================
# 3. BUILD CONTEXT
# ==========================================

def build_context(results):

    context_parts = []

    for index, result in enumerate(results, start=1):

        metadata = result.get("metadata", {})
        text = result.get("text", "")

        if not text:
            text = result.get("content", "")

        filename = metadata.get(
            "filename",
            metadata.get("source", "Unknown document")
        )

        category = metadata.get(
            "category",
            "General"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )

        context_parts.append(
            f"""
SOURCE {index}
Document: {filename}
Category: {category}
Page: {page}

Content:
{text}
"""
        )

    return "\n".join(context_parts)


# ==========================================
# 4. RESPONSE PARSING
# ==========================================

def extract_field(response_text, field):

    known_fields = [
        "STATUS",
        "FACTUAL SUPPORT",
        "DEFINITION COVERAGE",
        "QUESTION COVERAGE",
        "SUPPORTED",
        "MISSING",
        "REASON",
        "VERIFIED ANSWER",
        "VERIFICATION SUMMARY"
    ]

    next_headers = "|".join(
        re.escape(name) for name in known_fields
    )

    pattern = (
        rf"^[ \t]*(?:\*\*)?{re.escape(field)}"
        rf"(?:\*\*)?[ \t]*:[ \t]*(?:\*\*)?"
        rf"(.*?)"
        rf"(?=^[ \t]*(?:\*\*)?(?:{next_headers})"
        rf"(?:\*\*)?[ \t]*:|\Z)"
    )

    match = re.search(
        pattern,
        response_text,
        re.IGNORECASE | re.MULTILINE | re.DOTALL
    )

    if match:
        return match.group(1).strip()

    return "Not specified."


def extract_status(response_text):

    match = re.search(
        r"^[ \t]*(?:\*\*)?STATUS(?:\*\*)?[ \t]*:"
        r"[ \t]*(SUFFICIENT|PARTIAL|INSUFFICIENT)\b",
        response_text,
        re.IGNORECASE | re.MULTILINE
    )

    if match:
        return match.group(1).upper()

    return "INSUFFICIENT"


# ==========================================
# 5. EVIDENCE ASSESSMENT
# ==========================================

def check_context_sufficiency(question, context):

    if SHOW_DIAGNOSTICS:
        print("\nEvaluating retrieved evidence...", flush=True)

    prompt = f"""
You are an evidence assessment system for a cattle-health
and veterinary knowledge base.

Your task is to determine whether the retrieved document
excerpts contain enough evidence to answer the user's question.

USER QUESTION:
{question}

RETRIEVED DOCUMENT EXCERPTS:
{context}

Evaluate the evidence using these four dimensions:

1. FACTUAL SUPPORT
Identify the facts directly supported by the excerpts.

2. DEFINITION COVERAGE
If the question asks what something is, determine whether
the excerpts contain enough information to construct an
accurate definition.

A formal one-line definition does not need to appear
word-for-word in the documents.

You may combine multiple directly supported facts
to construct a definition.

Do not invent missing information.

3. QUESTION COVERAGE
Determine whether the available evidence addresses the
actual question and its important components.

For multi-part questions, check every requested component.

4. EVIDENCE GAPS
Identify important information that is genuinely missing.

Do not classify information as missing merely because
the documents use different wording.

CLASSIFICATION RULES:

SUFFICIENT:
The retrieved evidence adequately supports the main answer.

PARTIAL:
Some parts of the question are supported, but important
requested information is missing.

INSUFFICIENT:
The retrieved evidence does not contain enough relevant
information to answer reliably.

Additional rules:
- Do not use outside knowledge.
- Do not assume unsupported medical facts.
- Distinguish missing formal wording from missing facts.
- Do not mark a question insufficient simply because
  a formal definition is absent.
- For broad questions, provide the basic answer first.
- For specific questions, assess the requested details.

Return your assessment in EXACTLY this format:

STATUS: SUFFICIENT / PARTIAL / INSUFFICIENT

FACTUAL SUPPORT:
Your assessment.

DEFINITION COVERAGE:
Your assessment.

QUESTION COVERAGE:
Your assessment.

SUPPORTED:
- Supported fact 1
- Supported fact 2

MISSING:
- Missing information, or None.

REASON:
Explain your classification briefly.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict evidence evaluator. "
                    "Use only the supplied document excerpts. "
                    "Never invent evidence."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=1200,
        extra_body={
            "reasoning_effort": "low",
            "clear_thinking": True
        }
    )

    result = response.choices[0].message.content or ""

    evaluation = {
        "status": extract_status(result),
        "factual_support": extract_field(result, "FACTUAL SUPPORT"),
        "definition_coverage": extract_field(result, "DEFINITION COVERAGE"),
        "question_coverage": extract_field(result, "QUESTION COVERAGE"),
        "supported": extract_field(result, "SUPPORTED"),
        "missing": extract_field(result, "MISSING"),
        "reason": extract_field(result, "REASON")
    }

    return evaluation, response


# ==========================================
# 6. ANSWER GENERATION
# ==========================================

def generate_answer(question, context, evaluation):

    prompt = f"""
You are an AI cattle-health and veterinary knowledge assistant.

Answer the user's question using ONLY the supplied knowledge-base excerpts.

USER QUESTION:
{question}

EVIDENCE ASSESSMENT:
Status: {evaluation["status"]}

Factual support:
{evaluation["factual_support"]}

Definition coverage:
{evaluation["definition_coverage"]}

Question coverage:
{evaluation["question_coverage"]}

Supported information:
{evaluation["supported"]}

Missing information:
{evaluation["missing"]}

Retrieved document excerpts:
{context}

ANSWERING INSTRUCTIONS:

1. Answer the actual question directly and start with the main answer.
2. Combine related facts from the excerpts into one coherent explanation.
3. Use only information supported by the supplied excerpts.
4. Do not invent symptoms, dosages, treatments or veterinary recommendations.
5. Cite factual claims using the document filename and page number.
6. Keep simple answers concise and detailed answers sufficiently complete.
7. If evidence is PARTIAL, clearly explain what information is unavailable.
8. Include a veterinary consultation caveat when discussing diagnosis or treatment.
9. Write complete, grammatically correct sentences and complete paragraphs.
10. Explain how related facts connect instead of listing disconnected keywords.
11. Do not end a sentence or paragraph halfway through.
12. Finish the current explanation before ending the response.
13. Do not add a conclusion that merely repeats the entire answer.

If evidence is insufficient, do not fabricate an answer.
Return only the final answer.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful cattle-health assistant. Provide accurate, "
                "evidence-grounded answers in clear, complete sentences. "
                "Never intentionally leave an explanation unfinished."
            )
        },
        {"role": "user", "content": prompt}
    ]

    answer_parts = []
    answer_responses = []

    for attempt in range(MAX_ANSWER_CONTINUATIONS + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=3000,
            extra_body={
                "reasoning_effort": "low",
                "clear_thinking": True
            }
        )

        answer_responses.append(response)
        message = response.choices[0].message
        part = message.content or ""
        answer_parts.append(part)

        if response.choices[0].finish_reason != "length":
            break

        messages.append({"role": "assistant", "content": part})
        messages.append({
            "role": "user",
            "content": (
                "Continue from the exact point where you stopped. Do not "
                "repeat earlier text. Finish the incomplete explanation and "
                "end with a complete sentence. Use only the original evidence."
            )
        })

    answer = "\\n".join(
        part for part in answer_parts if part.strip()
    ).strip()

    return answer, answer_responses


# ==========================================
# 7. CLAIM-LEVEL VERIFICATION
# ==========================================

def verify_answer_grounding(
    question,
    context,
    evaluation,
    draft_answer
):

    if SHOW_DIAGNOSTICS:
        print("\nVerifying individual factual claims...", flush=True)

    prompt = f"""
You are a claim-level verification system
for a veterinary knowledge assistant.

Your task is to verify the draft answer against
the original retrieved document excerpts.

USER QUESTION:
{question}

EVIDENCE ASSESSMENT:
Status: {evaluation["status"]}

Supported:
{evaluation["supported"]}

Missing:
{evaluation["missing"]}

Reason:
{evaluation["reason"]}

ORIGINAL DOCUMENT EXCERPTS:
{context}

DRAFT ANSWER:
{draft_answer}

VERIFICATION INSTRUCTIONS:

1. Examine each factual claim in the draft answer.
2. Check whether the original excerpts support it.
3. Check whether the cited document and page are correct.
4. Remove unsupported claims.
5. Correct inaccurate or exaggerated claims.
6. Preserve important qualifications and limitations.
7. Do not introduce external knowledge.
8. Do not approve claims merely because the draft states them.
9. Information listed as missing must not be approved
   unless the original excerpts independently support it.
10. Multiple excerpts may collectively support a claim.
11. Do not remove valid claims just because their wording
    differs from the source.
12. Keep the final answer focused on the user's question.
13. Preserve complete, grammatically correct sentences.
14. Preserve the logical flow and useful explanations
    of the draft answer.
15. Do not turn a well-explained paragraph into
    disconnected bullet points.
16. Remove only unsupported information, not supported
    explanations or relevant details.

Return EXACTLY this format:

VERIFIED ANSWER:
The corrected final answer.

VERIFICATION SUMMARY:
Claims checked: [number]
Supported claims: [number]
Removed or corrected claims: [number]
Overall: VERIFIED / CORRECTED / REJECTED
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict factual verifier. "
                    "Never approve unsupported claims. "
                    "Preserve clear and complete sentence structure."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=3500,
        extra_body={
            "reasoning_effort": "low",
            "clear_thinking": True
        }
    )

    result = response.choices[0].message.content or ""

    if SHOW_DIAGNOSTICS:
        print("\nRaw verification response:")
        print(result)

    verified_answer = extract_field(
        result,
        "VERIFIED ANSWER"
    )

    verification_summary = extract_field(
        result,
        "VERIFICATION SUMMARY"
    )

    if (
        verified_answer == "Not specified."
        or verification_summary == "Not specified."
    ):

        print("\nWARNING: Verification response format was invalid.")

        return (
            None,
            "Verification failed: Expected response fields were missing.",
            response
        )

    valid_status = re.search(
        r"\b(VERIFIED|CORRECTED|REJECTED)\b",
        verification_summary,
        re.IGNORECASE
    )

    if not valid_status:

        return (
            None,
            "Verification failed: Invalid verification status.",
            response
        )

    status = valid_status.group(1).upper()

    if status == "REJECTED":

        return (
            None,
            verification_summary,
            response
        )

    return verified_answer, verification_summary, response


# ==========================================
# 8. TOKEN USAGE AND COST
# ==========================================

def display_usage(responses):

    total_input = 0
    total_output = 0

    for response in responses:

        if response is None:
            continue

        usage = getattr(response, "usage", None)

        if usage:
            total_input += usage.prompt_tokens
            total_output += usage.completion_tokens

    total_tokens = total_input + total_output

    input_cost = (
        total_input / 1_000_000
    ) * INPUT_PRICE

    output_cost = (
        total_output / 1_000_000
    ) * OUTPUT_PRICE

    total_cost = input_cost + output_cost

    print("\n========== TOKEN USAGE ==========")
    print(f"Input tokens: {total_input}")
    print(f"Output tokens: {total_output}")
    print(f"Total tokens: {total_tokens}")

    print("\n========== ESTIMATED API COST ==========")
    print(f"Input cost: ₹{input_cost:.8f}")
    print(f"Output cost: ₹{output_cost:.8f}")
    print(f"Total cost: ₹{total_cost:.8f}")

    print(
        "\nNote: This estimate covers the RAG model calls "
        "tracked here. Guardrails usage is not included."
    )


# ==========================================
# 9. PROCESSING TIME REPORT
# ==========================================

def display_timing(timings, total_start):

    total_time = time.perf_counter() - total_start

    print("\n========== PROCESSING TIME ==========")

    for stage, duration in timings.items():
        print(f"{stage:<24} {duration:>8.2f} seconds")

    print("-------------------------------------")
    print(f"{'TOTAL TIME':<24} {total_time:>8.2f} seconds")


# ==========================================
# 10. MAIN QUESTION PROCESSING
# ==========================================

def ask_question(question):

    total_start = time.perf_counter()

    timings = {
        "Guardrails": 0.0,
        "Retrieval": 0.0,
        "Evidence assessment": 0.0,
        "Answer generation": 0.0,
        "Claim verification": 0.0
    }
    responses = []

    # 1. Check whether the question belongs to the cattle-health domain.
    stage_start = time.perf_counter()
    allowed, rejection_message = check_guardrails(question)
    timings["Guardrails"] = time.perf_counter() - stage_start

    if not allowed:
        print("\\nCATTLE HEALTH ASSISTANT\\n")
        print(rejection_message)
        if SHOW_DIAGNOSTICS:
            display_timing(timings, total_start)
        return

    # Handle greetings directly.
    if question.strip().lower() in [
        "hello", "hi", "hey", "good morning",
        "good afternoon", "good evening"
    ]:
        print(
            "\\nCATTLE HEALTH ASSISTANT\\n"
            "Hello! I can help with cattle diseases, health, "
            "nutrition and dairy farming."
        )
        return

    # 2. Retrieve the best passages internally. Do not display chunks.
    stage_start = time.perf_counter()
    results = retrieve_context(question, top_k=5)
    timings["Retrieval"] = time.perf_counter() - stage_start

    if not results:
        print(
            "\\nCATTLE HEALTH ASSISTANT\\n"
            "I could not find relevant information in the local "
            "knowledge base to answer this question."
        )
        if SHOW_DIAGNOSTICS:
            display_timing(timings, total_start)
        return

    context = build_context(results)

    if not context.strip():
        print(
            "\\nCATTLE HEALTH ASSISTANT\\n"
            "The retrieved documents did not contain usable text."
        )
        if SHOW_DIAGNOSTICS:
            display_timing(timings, total_start)
        return

    # 3. Assess the combined evidence internally.
    stage_start = time.perf_counter()
    evaluation, evaluation_response = check_context_sufficiency(
        question,
        context
    )
    timings["Evidence assessment"] = time.perf_counter() - stage_start
    responses.append(evaluation_response)

    if SHOW_DIAGNOSTICS:
        print("\\n========== EVIDENCE ASSESSMENT ==========")
        print(f"Status: {evaluation['status']}")
        print(f"Factual Support: {evaluation['factual_support']}")
        print(f"Definition Coverage: {evaluation['definition_coverage']}")
        print(f"Question Coverage: {evaluation['question_coverage']}")
        print(f"Supported: {evaluation['supported']}")
        print(f"Missing: {evaluation['missing']}")
        print(f"Reason: {evaluation['reason']}")

    if evaluation["status"] == "INSUFFICIENT":
        print(
            "\\nCATTLE HEALTH ASSISTANT\\n"
            "I could not find sufficient supporting information in "
            "the local knowledge base to answer this question reliably."
        )
        if SHOW_DIAGNOSTICS:
            display_timing(timings, total_start)
            display_usage(responses)
        return

    # 4. Generate one synthesized answer from the retrieved passages.
    stage_start = time.perf_counter()
    draft_answer, answer_responses = generate_answer(
        question,
        context,
        evaluation
    )
    timings["Answer generation"] = time.perf_counter() - stage_start
    responses.extend(answer_responses)

    if not draft_answer.strip():
        print(
            "\\nCATTLE HEALTH ASSISTANT\\n"
            "I could not generate an answer from the available evidence."
        )
        if SHOW_DIAGNOSTICS:
            display_timing(timings, total_start)
            display_usage(responses)
        return

    # 5. Verify the answer before displaying it.
    stage_start = time.perf_counter()
    verified_answer, verification_summary, verification_response = (
        verify_answer_grounding(
            question,
            context,
            evaluation,
            draft_answer
        )
    )
    timings["Claim verification"] = time.perf_counter() - stage_start
    responses.append(verification_response)

    if SHOW_DIAGNOSTICS:
        print("\\n========== CLAIM VERIFICATION ==========")
        print(verification_summary)

    # Display only the final verified answer to the user.
    print("\\nCATTLE HEALTH ASSISTANT\\n")

    if verified_answer is not None and verified_answer.strip():
        print(verified_answer.strip())
    else:
        print(
            "I could not safely display the generated answer because "
            "claim verification did not complete successfully."
        )

    # Detailed technical metrics are hidden by default.
    if SHOW_DIAGNOSTICS:
        display_timing(timings, total_start)
        display_usage(responses)


# ==========================================
# 11. COMMAND LINE INTERFACE
# ==========================================

def main():

    print("\n========================================")
    print("       CATTLE HEALTH RAG ASSISTANT")
    print("========================================")

    print("Ask questions about:")
    print("- Cattle diseases")
    print("- Mastitis, FMD and LSD")
    print("- Body Condition Scoring")
    print("- Dairy nutrition and livestock management")

    print("\nCurrently using: Local PDF Knowledge Base")
    print("Internet search: Not enabled yet")
    print("Type 'exit' to quit.")

    while True:

        try:

            question = input("\nYou: ").strip()

            if question.lower() in ["exit", "quit"]:

                print("\nThank you for using the Cattle Health Assistant!")
                break

            if not question:
                print("Please enter a question.")
                continue

            ask_question(question)

        except KeyboardInterrupt:

            print("\n\nAssistant stopped.")
            break

        except Exception as error:

            print(f"\nAn error occurred: {error}")


if __name__ == "__main__":
    main()

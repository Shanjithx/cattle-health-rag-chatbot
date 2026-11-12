import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

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

# Keep False for normal user-facing operation.
# Set True temporarily when benchmarking the pipeline.
SHOW_DIAGNOSTICS = True

# One combined GLM request handles evidence assessment, KB answer, and
# conditional fallback. If truncated, the response is withheld safely.
COMBINED_MAX_TOKENS = 3000
REQUEST_TIMEOUT_SECONDS = 60

# The local PDF pipeline has no live literature-search capability. Queries
# explicitly asking for the latest/current research must not receive an
# unverified answer from pretrained model knowledge.
RECENCY_REQUEST = re.compile(
    r"\b(?:latest|most recent|recent research|current research|"
    r"up[- ]to[- ]date|newest|state[- ]of[- ]the[- ]art|"
    r"recent studies|latest studies)\b",
    re.IGNORECASE,
)

# Pricing per 1 million tokens (INR). Update these if your provider's
# current pricing changes.
INPUT_PRICE = 7.11
OUTPUT_PRICE = 23.69

if not API_KEY:
    raise ValueError("JARVIS_API_KEY is missing from .env")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=REQUEST_TIMEOUT_SECONDS,
)


# ==========================================
# 2. SMALL UTILITIES
# ==========================================

def diagnostic(message: str) -> None:
    if SHOW_DIAGNOSTICS:
        print(message, flush=True)


def record_stage(
    timings: Dict[str, float],
    stage: str,
    start_time: float,
) -> None:
    timings[stage] = time.perf_counter() - start_time


def extract_section(
    response_text: str,
    field: str,
    following_fields: List[str],
) -> Optional[str]:
    """Extract a labelled section from the model's plain-text response."""
    if following_fields:
        next_headers = "|".join(re.escape(name) for name in following_fields)
        end_pattern = (
            rf"(?=^[ \t]*(?:\*\*)?(?:{next_headers})"
            rf"(?:\*\*)?[ \t]*:|\Z)"
        )
    else:
        end_pattern = r"(?=\Z)"

    pattern = (
        rf"^[ \t]*(?:\*\*)?{re.escape(field)}(?:\*\*)?[ \t]*:"
        rf"[ \t]*(?:\*\*)?(.*?)"
        rf"{end_pattern}"
    )
    match = re.search(
        pattern,
        response_text or "",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    return match.group(1).strip() if match else None


def extract_status(response_text: str) -> str:
    match = re.search(
        r"^[ \t]*(?:\*\*)?STATUS(?:\*\*)?[ \t]*:"
        r"[ \t]*(SUFFICIENT|PARTIAL|INSUFFICIENT)\b",
        response_text or "",
        re.IGNORECASE | re.MULTILINE,
    )
    return match.group(1).upper() if match else "INSUFFICIENT"


def extract_confidence(response_text: str) -> str:
    match = re.search(
        r"^[ \t]*(?:\*\*)?CONFIDENCE(?:\*\*)?[ \t]*:"
        r"[ \t]*(HIGH|LOW)\b",
        response_text or "",
        re.IGNORECASE | re.MULTILINE,
    )
    return match.group(1).upper() if match else "LOW"


def get_response_text(response: Any) -> str:
    if not response or not getattr(response, "choices", None):
        return ""
    message = response.choices[0].message
    return (getattr(message, "content", None) or "").strip()


def get_finish_reason(response: Any) -> str:
    try:
        return str(response.choices[0].finish_reason or "")
    except (AttributeError, IndexError, TypeError):
        return ""


# ==========================================
# 3. INDIVIDUALIZED MEDICATION SAFETY GATE
# ==========================================

MEDICATION_TERMS = re.compile(
    r"\b(?:antibiotic|antibiotics|medicine|medication|drug|injection|"
    r"oxytetracycline|ceftiofur|amikacin|gentamicin|enrofloxacin|"
    r"penicillin|intramammary|dosage|dose|prescription)\b",
    re.IGNORECASE,
)

DOSING_ACTIONS = re.compile(
    r"\b(?:dose|dosage|how much|how many|administer|give|inject|"
    r"prescribe|start|use|treat(?:ment)?|recommend)\b",
    re.IGNORECASE,
)

INDIVIDUAL_ANIMAL_CUES = re.compile(
    r"\b(?:my cow|my cattle|this cow|this animal|the cow|a cow|"
    r"cow weighing|weighing\s+\d+(?:\.\d+)?\s*(?:kg|kilograms?)|"
    r"\d+(?:\.\d+)?\s*(?:kg|kilograms?)\s+(?:cow|cattle|calf)|"
    r"450\s*kg|individual animal|severe mastitis|clinical mastitis|"
    r"mastitis in (?:my|this|the) cow)\b",
    re.IGNORECASE,
)

PERSONALIZED_TREATMENT_PATTERNS = [
    re.compile(
        r"\b(?:what|which|how much|how many).{0,100}"
        r"\b(?:antibiotic|medicine|medication|drug|dose|dosage|prescription)"
        r".{0,120}\b(?:my cow|this cow|a cow|the cow|cattle|calf)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:my cow|this cow|a cow|the cow|cattle|calf)\b.{0,120}"
        r"\b(?:dose|dosage|antibiotic|medicine|medication|drug|"
        r"treat|treatment|inject|administer|prescribe)\b",
        re.IGNORECASE | re.DOTALL,
    ),
]


def is_individualized_medication_request(question: str) -> bool:
    """
    Detect requests for a medication choice, dose, or treatment plan
    for a particular animal. This deterministic gate prevents retrieved
    general dosage tables from being presented as an individual prescription.

    General educational questions about drug information in a document are
    not blocked unless they contain individualized-animal cues.
    """
    text = (question or "").strip()
    if not text:
        return False

    has_medication_term = bool(MEDICATION_TERMS.search(text))
    has_dosing_action = bool(DOSING_ACTIONS.search(text))
    has_individual_cue = bool(INDIVIDUAL_ANIMAL_CUES.search(text))

    if has_medication_term and has_dosing_action and has_individual_cue:
        return True

    return any(pattern.search(text) for pattern in PERSONALIZED_TREATMENT_PATTERNS)


def medication_safety_response() -> str:
    return (
        "I cannot recommend an antibiotic, exact dosage, or individualized "
        "treatment plan for a specific cow. Severe mastitis can become "
        "life-threatening and requires urgent veterinary assessment. "
        "Please contact a qualified veterinarian immediately. The veterinarian "
        "must select the appropriate treatment based on the cow's clinical "
        "condition, likely or confirmed pathogen, approved medicines, and "
        "milk and meat withdrawal requirements. Do not use a general dosage "
        "listed for another infection as a mastitis prescription."
    )


# ==========================================
# 3. HYBRID RETRIEVAL WITH RERANKING
# ==========================================

def retrieve_context(question: str, top_k: int = 5) -> List[dict]:
    diagnostic("\nSearching the cattle knowledge base...")

    results = hybrid_search(
        question,
        top_k=top_k,
        verbose=SHOW_DIAGNOSTICS,
    )

    diagnostic(f"\nRetrieved {len(results)} reranked passages.")
    return results or []


# ==========================================
# 4. BUILD CONTEXT
# ==========================================

def build_context(results: List[dict]) -> str:
    context_parts = []

    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata", {}) or {}
        text = result.get("text") or result.get("content") or ""

        if not text.strip():
            continue

        filename = metadata.get(
            "filename",
            metadata.get("source", "Unknown document"),
        )
        category = metadata.get("category", "General")
        page = metadata.get("page", "Unknown")

        context_parts.append(
            f"""
SOURCE {index}
Document: {filename}
Category: {category}
Page: {page}

Content:
{text.strip()}
""".strip()
        )

    return "\n\n---\n\n".join(context_parts)


# ==========================================
# 5. SINGLE-CALL EVIDENCE + ANSWER + FALLBACK
# ==========================================

def assess_and_generate_complete_answer(
    question: str,
    context: str,
) -> Tuple[Dict[str, str], Any]:
    """
    One GLM request performs evidence assessment, writes the PDF-grounded
    answer, assesses whether missing information can safely be supplemented,
    and conditionally writes that supplementary information.

    The model's grounding self-check is not independent clinical verification.
    """

    prompt = f"""
You are a careful cattle-health and veterinary educational assistant.
Prioritise factual accuracy, species-specific evidence, and animal safety.

USER QUESTION:
{question}

RETRIEVED LOCAL KNOWLEDGE-BASE EXCERPTS:
{context if context.strip() else "No usable PDF excerpts were retrieved."}

TASK A — ASSESS THE LOCAL EVIDENCE

Assess the excerpts against the actual question:
1. FACTUAL SUPPORT: identify facts directly supported by the excerpts.
2. DEFINITION COVERAGE: if a definition is requested, assess whether the
   excerpts provide enough information to construct it accurately.
3. QUESTION COVERAGE: assess every important part of a multi-part question.
4. EVIDENCE GAPS: identify important requested information not supported.

Choose exactly one status:
- SUFFICIENT: excerpts adequately support the requested answer.
- PARTIAL: some important parts are supported, but others are missing.
- INSUFFICIENT: excerpts do not contain enough relevant evidence.

TASK B — WRITE THE PDF-GROUNDED ANSWER

- Use ONLY facts supported by the supplied excerpts.
- Cite factual claims as (Document: filename, page: number), using the
  exact document names and page numbers supplied.
- For PARTIAL, answer only supported parts and identify important gaps.
- For INSUFFICIENT, write NONE.
- Do not invent citations, symptoms, dosages, treatments, thresholds,
  or veterinary recommendations.
- Do not use evidence about another species as cattle-specific evidence.
- Preserve qualifications and distinguish association from causation.
- Do not diagnose an individual animal or provide prescriptions or
  individualized treatment instructions.
- Check material claims against the excerpts before responding.

TASK C — CONDITIONAL GENERAL-KNOWLEDGE FALLBACK

If status is SUFFICIENT:
- Set FALLBACK CONFIDENCE to NOT_NEEDED.
- Set FALLBACK REASON to "The local evidence sufficiently answers the question."
- Set SUPPLEMENTARY ANSWER to NONE.

If status is PARTIAL:
- Consider ONLY the important information identified as missing.
- If status is INSUFFICIENT:
  consider whether the whole question can be answered with reliable,
  well-established general educational knowledge.
- Use general pretrained knowledge only when it is reliable, broadly
  accepted, and can be stated without guessing.
- Do not pretend general knowledge came from the PDFs.
- Choose HIGH only if all supplementary claims can be stated reliably.
  Choose LOW if any important requested part is uncertain, disputed,
  case-specific, requires examination/testing, or depends on current/local
  information that cannot be verified.
- If the user asks for the latest, most recent, current, newest, or
  up-to-date research, choose LOW unless dated, relevant research evidence
  is explicitly present in the supplied excerpts. This application has no
  live literature-search capability; pretrained knowledge cannot establish
  what is latest.
- Do not provide exact BCS score ranges, target scores, cutoffs, or numerical
  thresholds from general knowledge unless those exact values are supported
  by the supplied excerpts. You may explain qualitative relationships
  without inventing numeric targets.
- Do not describe broad background knowledge as a recent research finding.
- If HIGH, write concise supplementary information addressing only the gap.
- If LOW, write NONE.
- Do not repeat the PDF-grounded answer unnecessarily.

MANDATORY VETERINARY SAFETY:
- Never provide individualized medicine selection, dosage, route,
  frequency, duration, prescription, or treatment instructions.
- Do not diagnose an individual animal remotely.
- Do not invent withdrawal periods, drug protocols, clinical findings,
  numerical thresholds, disease-risk percentages, or quantitative effects.
- Do not invent biological mechanisms to explain an association.
- For diagnosis, treatment, severe symptoms, or emergencies, recommend
  assessment by a qualified veterinarian.
- Distinguish correlation from causation.
- If uncertain, abstain rather than guess.

ANSWER STYLE:
- Be direct, concise, and structured.
- Include only information needed to answer the question.
- Do not expose internal reasoning, confidence labels, or retrieval scores
  in the user-facing answer.
- The diagnostic fields below are for the application to parse; the CLI
  should display only the composed answer unless diagnostics are enabled.

Return EXACTLY these headings, each on its own line:

STATUS: SUFFICIENT, PARTIAL, or INSUFFICIENT

FACTUAL SUPPORT:
Brief assessment.

DEFINITION COVERAGE:
Brief assessment.

QUESTION COVERAGE:
Brief assessment.

SUPPORTED:
- Supported fact
(or NONE)

MISSING:
- Important missing information
(or NONE)

REASON:
Brief explanation.

KB ANSWER:
Complete answer supported by the PDFs, or NONE.

FALLBACK CONFIDENCE: HIGH, LOW, or NOT_NEEDED

FALLBACK REASON:
Brief explanation.

SUPPLEMENTARY ANSWER:
Concise general educational information, or NONE.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict, cautious cattle-health assistant. "
                    "Ground PDF claims in supplied excerpts, distinguish "
                    "general knowledge from source evidence, and never guess."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=COMBINED_MAX_TOKENS,
        extra_body={
            "reasoning_effort": "low",
            "clear_thinking": True,
        },
    )

    raw = get_response_text(response)

    fields = [
        "STATUS",
        "FACTUAL SUPPORT",
        "DEFINITION COVERAGE",
        "QUESTION COVERAGE",
        "SUPPORTED",
        "MISSING",
        "REASON",
        "KB ANSWER",
        "FALLBACK CONFIDENCE",
        "FALLBACK REASON",
        "SUPPLEMENTARY ANSWER",
    ]

    confidence_match = re.search(
        r"^[ \t]*(?:\*\*)?FALLBACK CONFIDENCE(?:\*\*)?[ \t]*:"
        r"[ \t]*(HIGH|LOW|NOT_NEEDED)\b",
        raw,
        re.IGNORECASE | re.MULTILINE,
    )
    fallback_confidence = (
        confidence_match.group(1).upper() if confidence_match else "LOW"
    )

    evaluation = {
        "status": extract_status(raw),
        "factual_support": extract_section(raw, "FACTUAL SUPPORT", fields[2:]) or "Not specified.",
        "definition_coverage": extract_section(raw, "DEFINITION COVERAGE", fields[3:]) or "Not specified.",
        "question_coverage": extract_section(raw, "QUESTION COVERAGE", fields[4:]) or "Not specified.",
        "supported": extract_section(raw, "SUPPORTED", fields[5:]) or "None.",
        "missing": extract_section(raw, "MISSING", fields[6:]) or "Not specified.",
        "reason": extract_section(raw, "REASON", fields[7:]) or "Not specified.",
        "kb_answer": extract_section(raw, "KB ANSWER", fields[8:]) or "",
        "fallback_confidence": fallback_confidence,
        "fallback_reason": extract_section(raw, "FALLBACK REASON", fields[10:]) or "Not specified.",
        "fallback_answer": extract_section(raw, "SUPPLEMENTARY ANSWER", []) or "",
    }

    required_fields = (
        evaluation["factual_support"],
        evaluation["definition_coverage"],
        evaluation["question_coverage"],
        evaluation["supported"],
        evaluation["missing"],
        evaluation["reason"],
        evaluation["fallback_reason"],
    )

    # Fail closed on truncation or malformed structured output.
    if (
        get_finish_reason(response) == "length"
        or any(value == "Not specified." for value in required_fields)
    ):
        evaluation["status"] = "INSUFFICIENT"
        evaluation["kb_answer"] = ""
        evaluation["fallback_confidence"] = "LOW"
        evaluation["fallback_answer"] = ""
        evaluation["reason"] = (
            "The combined response was truncated or malformed; "
            "the generated answer was withheld."
        )
        return evaluation, response

    if evaluation["kb_answer"].strip().upper() == "NONE":
        evaluation["kb_answer"] = ""

    if evaluation["fallback_answer"].strip().upper() == "NONE":
        evaluation["fallback_answer"] = ""

    # Enforce recency limitation outside the model prompt as well. A model
    # confidence label cannot make pretrained knowledge a verified latest study.
    evaluation["recency_sensitive"] = bool(RECENCY_REQUEST.search(question or ""))
    if evaluation["recency_sensitive"]:
        evaluation["fallback_confidence"] = "LOW"
        evaluation["fallback_answer"] = ""
        evaluation["fallback_reason"] = (
            "The question requests current research, but this application "
            "has no live literature-search source to verify recency."
        )

    # Guard against unsupported numeric BCS scales/targets in general-knowledge
    # fallback text. The PDF-grounded answer is retained with its citations.
    bcs_question = bool(re.search(r"\b(?:BCS|body condition score|body condition scoring)\b", question or "", re.I))
    bcs_numeric_claim = bool(re.search(
        r"(?:\bBCS\b|body condition\w*)[^.\n]{0,100}\b\d+(?:\.\d+)?\b"
        r"|\b\d+(?:\.\d+)?\b[^.\n]{0,100}(?:\bBCS\b|body condition\w*)",
        evaluation["fallback_answer"],
        re.IGNORECASE,
    ))
    if bcs_question and bcs_numeric_claim:
        evaluation["fallback_confidence"] = "LOW"
        evaluation["fallback_answer"] = ""
        evaluation["fallback_reason"] = (
            "The supplementary answer included numerical BCS values or "
            "thresholds that were not verified by the supplied PDFs."
        )

    if evaluation["status"] == "SUFFICIENT":
        evaluation["fallback_confidence"] = "NOT_NEEDED"
        evaluation["fallback_answer"] = ""
        if not evaluation["kb_answer"]:
            evaluation["status"] = "INSUFFICIENT"
            evaluation["reason"] = (
                "The model classified the evidence as sufficient but "
                "did not return a usable KB answer."
            )

    elif evaluation["status"] == "PARTIAL":
        if not evaluation["kb_answer"]:
            evaluation["status"] = "INSUFFICIENT"
        if (
            evaluation["fallback_confidence"] != "HIGH"
            or not evaluation["fallback_answer"]
        ):
            evaluation["fallback_confidence"] = "LOW"
            evaluation["fallback_answer"] = ""

    elif evaluation["status"] == "INSUFFICIENT":
        evaluation["kb_answer"] = ""
        if (
            evaluation["fallback_confidence"] != "HIGH"
            or not evaluation["fallback_answer"]
        ):
            evaluation["fallback_confidence"] = "LOW"
            evaluation["fallback_answer"] = ""

    else:
        evaluation["status"] = "INSUFFICIENT"
        evaluation["kb_answer"] = ""
        evaluation["fallback_confidence"] = "LOW"
        evaluation["fallback_answer"] = ""

    return evaluation, response


# ==========================================
# 7. TOKEN USAGE AND COST (DIAGNOSTICS ONLY)
# ==========================================

def display_usage(responses: List[Any]) -> None:
    total_input = 0
    total_output = 0

    for response in responses:
        if response is None:
            continue

        usage = getattr(response, "usage", None)
        if usage:
            total_input += getattr(usage, "prompt_tokens", 0) or 0
            total_output += getattr(usage, "completion_tokens", 0) or 0

    total_tokens = total_input + total_output
    input_cost = (total_input / 1_000_000) * INPUT_PRICE
    output_cost = (total_output / 1_000_000) * OUTPUT_PRICE
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
        "\nNote: This estimate covers the RAG model calls tracked here. "
        "Guardrails usage is not included."
    )


def display_timing(
    timings: Dict[str, float],
    total_start: float,
) -> None:
    total_time = time.perf_counter() - total_start

    print("\n========== PROCESSING TIME ==========")
    for stage, duration in timings.items():
        print(f"{stage:<24} {duration:>8.2f} seconds")

    print("-------------------------------------")
    print(f"{'TOTAL TIME':<24} {total_time:>8.2f} seconds")


# ==========================================
# 8. FINAL ANSWER FORMATTING
# ==========================================

def compose_final_answer(
    status: str,
    kb_answer: str,
    fallback_answer: str,
    fallback_confidence: str,
    recency_sensitive: bool = False,
) -> str:
    """Build the user-facing answer without exposing internal diagnostics."""

    if status == "SUFFICIENT":
        return kb_answer.strip()

    if status == "PARTIAL":
        parts = []

        if kb_answer.strip():
            parts.append(kb_answer.strip())

        if fallback_confidence == "HIGH" and fallback_answer.strip():
            parts.append(
                "Additional general information (not supported by the "
                "local PDF knowledge base):\n\n"
                + fallback_answer.strip()
            )
        elif recency_sensitive:
            parts.append(
                "The local documents do not contain the requested recent "
                "research details. Internet/literature search is not enabled, "
                "so I cannot verify which findings are currently the latest. "
                "I have therefore not filled this gap with unverified model knowledge."
            )
        elif not kb_answer.strip():
            parts.append(
                "The local documents did not provide enough verified "
                "information to answer this safely. Please consult a "
                "qualified veterinarian."
            )
        else:
            parts.append(
                "The local documents do not cover all parts of your "
                "question, and reliable supplementary information could "
                "not be established. Please consult a qualified veterinarian."
            )

        return "\n\n".join(parts)

    if fallback_confidence == "HIGH" and fallback_answer.strip():
        return (
            fallback_answer.strip()
            + "\n\n"
            "Note: This is general educational information from the model's "
            "pretrained knowledge, not information supported by the local PDFs "
            "or an individual veterinary assessment."
        )

    if recency_sensitive:
        return (
            "The local documents do not contain the requested recent research "
            "details. Internet/literature search is not enabled, so I cannot "
            "verify which findings are currently the latest. I have therefore "
            "not filled this gap with unverified model knowledge."
        )

    return (
        "Not enough reliable information is available in the local "
        "knowledge base or from the model's knowledge to answer this safely. "
        "Please consult a qualified veterinarian."
    )


# ==========================================
# 9. MAIN QUESTION PROCESSING
# ==========================================

def ask_question(question: str) -> None:
    total_start = time.perf_counter()

    timings = {
        "Guardrails": 0.0,
        "Retrieval": 0.0,
        "Combined GLM request": 0.0,
    }
    responses = []

    # 1. Domain and safety guardrails.
    stage_start = time.perf_counter()
    allowed, rejection_message = check_guardrails(question)
    record_stage(timings, "Guardrails", stage_start)

    if not allowed:
        print("\nCATTLE HEALTH ASSISTANT\n")
        print(rejection_message)
        if SHOW_DIAGNOSTICS:
            display_timing(timings, total_start)
        return

    # Block individualized veterinary medication requests before retrieval.
    if is_individualized_medication_request(question):
        print("\nCATTLE HEALTH ASSISTANT\n")
        print(medication_safety_response())
        if SHOW_DIAGNOSTICS:
            display_timing(timings, total_start)
        return

    # Handle simple greetings without retrieval or generation.
    greetings = {
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
    }
    if question.strip().lower() in greetings:
        print(
            "\nCATTLE HEALTH ASSISTANT\n"
            "Hello! I can help with cattle diseases, health, "
            "nutrition and dairy farming."
        )
        return

    # 2. Retrieve passages from the local PDF knowledge base.
    stage_start = time.perf_counter()
    results = retrieve_context(question, top_k=5)
    record_stage(timings, "Retrieval", stage_start)
    context = build_context(results) if results else ""

    # 3. One GLM request: evidence assessment + KB answer + conditional fallback.
    stage_start = time.perf_counter()
    evaluation, response = assess_and_generate_complete_answer(
        question,
        context,
    )
    record_stage(timings, "Combined GLM request", stage_start)
    responses.append(response)

    if SHOW_DIAGNOSTICS:
        print("\n========== INTERNAL EVIDENCE ASSESSMENT ==========")
        print(f"Status: {evaluation['status']}")
        print(f"Factual support: {evaluation['factual_support']}")
        print(f"Definition coverage: {evaluation['definition_coverage']}")
        print(f"Question coverage: {evaluation['question_coverage']}")
        print(f"Supported: {evaluation['supported']}")
        print(f"Missing: {evaluation['missing']}")
        print(f"Reason: {evaluation['reason']}")

        if evaluation["status"] != "SUFFICIENT":
            print("\n========== INTERNAL FALLBACK ASSESSMENT ==========")
            print(f"Confidence: {evaluation['fallback_confidence']}")
            print(f"Reason: {evaluation['fallback_reason']}")

    final_answer = compose_final_answer(
        status=evaluation["status"],
        kb_answer=evaluation["kb_answer"],
        fallback_answer=evaluation["fallback_answer"],
        fallback_confidence=evaluation["fallback_confidence"],
        recency_sensitive=evaluation.get("recency_sensitive", False),
    )

    print("\nCATTLE HEALTH ASSISTANT\n")
    print(final_answer)

    if SHOW_DIAGNOSTICS:
        display_timing(timings, total_start)
        display_usage(responses)


# ==========================================
# 10. COMMAND LINE INTERFACE
# ==========================================

def main() -> None:
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

            if question.lower() in ("exit", "quit"):
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
            # Do not expose stack traces, API payloads, or secrets to users.
            print(
                "\nThe assistant encountered an error while processing "
                "your question. Please try again."
            )
            if SHOW_DIAGNOSTICS:
                print(f"Diagnostic error: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()

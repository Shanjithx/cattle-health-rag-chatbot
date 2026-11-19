import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Import your core backend modules
from guardrails import check_guardrails
from router import route_query
from erp_tools import execute_supabase_tool
from rag_optimized import (
    ask_question, 
    assess_and_generate_complete_answer, 
    retrieve_context, 
    build_context, 
    compose_final_answer, 
    is_individualized_medication_request, 
    medication_safety_response
)

load_dotenv()

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "Cattle ERP & Hybrid RAG Backend"}), 200


@app.route("/chat", methods=["POST"])
def chat():
    """
    Main unified endpoint for the farmer assistant chatbot.
    Payload JSON:
    {
        "message": "What is my farm's milk yield today?",
        "farm_id": 1
    }
    """
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    farm_id = data.get("farm_id", 1)  # Default multi-tenant farm ID scoping

    if not message:
        return jsonify({"error": "Message content is required."}), 400

    # --------------------------------------------------
    # Step 1: Input Validation & Security Guardrails
    # --------------------------------------------------
    allowed, rejection_message = check_guardrails(message)
    if not allowed:
        return jsonify({
            "status": "blocked",
            "intent": "NONE",
            "response": rejection_message
        }), 200

    # Handle individualized medication safety gate for RAG queries
    if is_individualized_medication_request(message):
        return jsonify({
            "status": "success",
            "intent": "RAG",
            "response": medication_safety_response()
        }), 200

    # Handle simple greetings instantly
    greetings = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening"}
    if message.lower() in greetings:
        return jsonify({
            "status": "success",
            "intent": "CONVERSATIONAL",
            "response": "Hello! I can help with your farm account data, milk records, inventory, and cattle health questions."
        }), 200

    # --------------------------------------------------
    # Step 2: Adaptive Intent Routing (ERP vs RAG)
    # --------------------------------------------------
    intent = route_query(message)

    # --------------------------------------------------
    # Step 3: Branching Execution
    # --------------------------------------------------
    if intent == "ERP":
        # Route directly to schema-aware Supabase SQL execution engine
        response_text = execute_supabase_tool(message, farm_id=farm_id)
        return jsonify({
            "status": "success",
            "intent": "ERP",
            "response": response_text
        }), 200

    else:
        # Route to Hybrid RAG pipeline (pgvector + BM25 + RRF + Cross-Encoder + GLM-5.3 Flash)
        try:
            results = retrieve_context(message, top_k=5)
            context = build_context(results) if results else ""
            
            evaluation, _ = assess_and_generate_complete_answer(message, context)
            
            final_answer = compose_final_answer(
                status=evaluation["status"],
                kb_answer=evaluation["kb_answer"],
                fallback_answer=evaluation["fallback_answer"],
                fallback_confidence=evaluation["fallback_confidence"],
                recency_sensitive=evaluation.get("recency_sensitive", False)
            )

            return jsonify({
                "status": "success",
                "intent": "RAG",
                "evaluation_status": evaluation["status"],
                "response": final_answer
            }), 200

        except Exception as e:
            print(f"RAG Pipeline Error: {e}")
            return jsonify({
                "status": "error",
                "intent": "RAG",
                "response": "An error occurred while querying the veterinary knowledge base. Please try again."
            }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
import os
from dotenv import load_dotenv

# Import your core backend modules
from guardrails import check_guardrails
from router import route_query
from erp_tools import execute_supabase_tool
from rag_optimized import (
    retrieve_context, 
    build_context, 
    assess_and_generate_complete_answer, 
    compose_final_answer, 
    is_individualized_medication_request, 
    medication_safety_response
)

load_dotenv()

def run_cli_bot():
    print("=" * 60)
    print("        CATTLE HEALTH & ERP RAGUL CLI CHATBOT        ")
    print("=" * 60)
    print("Type your question below (or type 'exit' or 'quit' to stop).\n")
    
    # Default multi-tenant farm ID for local testing
    farm_id = 1  

    while True:
        try:
            message = input("\nFarmer / User: ").strip()
            if not message:
                continue
            if message.lower() in ["exit", "quit"]:
                print("\nExiting chatbot. Goodbye!")
                break

            # --------------------------------------------------
            # Step 1: Input Validation & Security Guardrails
            # --------------------------------------------------
            allowed, rejection_message = check_guardrails(message)
            if not allowed:
                print(f"\nBot [Blocked]: {rejection_message}")
                continue

            # Medication safety gate
            if is_individualized_medication_request(message):
                print(f"\nBot [Safety Notice]: {medication_safety_response()}")
                continue

            # Simple greetings
            greetings = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening"}
            if message.lower() in greetings:
                print("\nBot: Hello! I can help with your farm account data, milk records, inventory, and cattle health questions.")
                continue

            # --------------------------------------------------
            # Step 2: Adaptive Intent Routing (ERP vs RAG)
            # --------------------------------------------------
            intent = route_query(message)
            print(f"[Debug] Detected Intent: {intent}")

            # --------------------------------------------------
            # Step 3: Branching Execution
            # --------------------------------------------------
            if intent == "ERP":
                response_text = execute_supabase_tool(message, farm_id=farm_id)
                print(f"\nBot [ERP Tool Result]:\n{response_text}")
            else:
                # Hybrid RAG pipeline execution
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

                print(f"\nBot [RAG Knowledge Base Answer]:\n{final_answer}")

        except Exception as e:
            print(f"\n[Error]: {e}")

if __name__ == "__main__":
    run_cli_bot()
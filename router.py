import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("JARVIS_API_KEY")
base_url = os.getenv("JARVIS_BASE_URL")
model = os.getenv("JARVIS_MODEL", "glm-5.3-flash")

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=15.0
)

def route_query(question: str) -> str:
    """
     adaptively classifies user queries into:
    - 'ERP': For live, farm-specific database queries (milk, inventory, cow records).
    - 'RAG': For general veterinary, disease, nutrition, or document-based knowledge.
    """
    system_prompt = """
You are an intent router for a cattle farming assistant ERP platform.
Classify the user's question into one of two categories:

1. ERP: Questions asking about specific farm data, live metrics, logs, or records belonging to their farm. 
   Examples: 
   - "How much milk did I produce today?"
   - "Which cow gave the least milk in the past 10 days?"
   - "Show my feed inventory."
   - "What is the status of cow #12's medical record?"

2. RAG: Questions asking for general veterinary knowledge, disease facts, treatment guidelines, symptoms, nutrition, or educational information.
   Examples:
   - "What are the symptoms of mastitis?"
   - "How do I calculate Body Condition Score?"
   - "What causes Foot and Mouth Disease?"

Output ONLY one word:
ERP
or
RAG
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0,
            max_tokens=10,
            extra_body={
                "reasoning_effort": "low",
                "clear_thinking": True
            }
        )
        
        result = response.choices[0].message.content
        if not result:
            return "RAG"  # Default safe fallback
            
        intent = result.strip().upper()
        return "ERP" if "ERP" in intent else "RAG"
        
    except Exception as e:
        print(f"Routing classification error: {e}")
        return "RAG"  # Fail safe to veterinary knowledge base
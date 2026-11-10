
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

api_key = os.getenv("JARVIS_API_KEY")
base_url = os.getenv("JARVIS_BASE_URL")
model = os.getenv("JARVIS_MODEL")

if not api_key or not base_url or not model:
    raise ValueError("Check your .env configuration.")

# Connect to Jarvis Labs
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

print("Connecting to Jarvis Labs...")

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a cattle health assistant. "
                    "Answer directly and provide only the final answer."
                )
            },
            {
                "role": "user",
                "content": (
                    "Explain mastitis in cattle. "
                    "Include its definition and three common symptoms."
                )
            }
        ],
        temperature=0.3,
        max_tokens=1000,
        extra_body={
            "reasoning_effort": "low",
            "clear_thinking": True
        }
    )

    choice = response.choices[0]
    message = choice.message

    print("\n========== MODEL RESPONSE ==========")
    print("Finish reason:", choice.finish_reason)

    if message.content:
        print("\nMastitis Explanation:\n")
        print(message.content)
    else:
        print("\nNo final answer returned.")
        print(message.model_dump())

    # Token usage
    usage = response.usage

    if usage:
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        # Pricing from your Jarvis Labs screenshot
        input_price = 7.11
        output_price = 23.69

        input_cost = input_tokens / 1_000_000 * input_price
        output_cost = output_tokens / 1_000_000 * output_price

        print("\n========== TOKEN USAGE ==========")
        print("Input tokens:", input_tokens)
        print("Output tokens:", output_tokens)
        print("Total tokens:", total_tokens)

        print("\n========== ESTIMATED COST ==========")
        print(f"Input cost: ₹{input_cost:.8f}")
        print(f"Output cost: ₹{output_cost:.8f}")
        print(f"Total cost: ₹{input_cost + output_cost:.8f}")

    print("\nAPI test completed!")

except Exception as error:
    print("\nAPI ERROR:", type(error).__name__, str(error))

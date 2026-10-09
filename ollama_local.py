import pandas as pd
import ollama
from pathlib import Path
from datetime import datetime

# Dataset folder
DATASET = Path(__file__).resolve().parent / "dataset"
DATASET.mkdir(exist_ok=True)

# Load prompts
prompts = pd.read_csv(DATASET / "prompts.csv")

model = "llama3.2"
responses = []
comparison = []

# Conversation history
with open(DATASET / "conversation_history.txt", "w",
          encoding="utf-8") as history:

    history.write("WEEK 8 - LLM LOCAL INFERENCE\n")
    history.write(f"Model: {model}\n")
    history.write(f"Date: {datetime.now()}\n")
    history.write("=" * 60 + "\n\n")

    for index, row in prompts.iterrows():
        prompt = row["prompt"]

        print(f"Processing prompt {index + 1}...")

        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        answer = response["message"]["content"]

        # Store LLM responses
        responses.append({
            "prompt_id": row["prompt_id"],
            "domain": row["domain"],
            "prompt": prompt,
            "response": answer
        })

        # Store comparison data
        comparison.append({
            "prompt_id": row["prompt_id"],
            "domain": row["domain"],
            "prompt": prompt,
            "model": model,
            "response": answer,
            "response_length": len(answer)
        })

        # Store conversation history
        history.write(f"Prompt ID: {row['prompt_id']}\n")
        history.write(f"Domain: {row['domain']}\n")
        history.write(f"Prompt: {prompt}\n")
        history.write(f"Response: {answer}\n")
        history.write("-" * 60 + "\n\n")

# Save CSV files
pd.DataFrame(responses).to_csv(
    DATASET / "llm_responses.csv", index=False
)

pd.DataFrame(comparison).to_csv(
    DATASET / "response_comparison.csv", index=False
)

print("\nAll files saved successfully!")
print("1. dataset/llm_responses.csv")
print("2. dataset/response_comparison.csv")
print("3. dataset/conversation_history.txt")
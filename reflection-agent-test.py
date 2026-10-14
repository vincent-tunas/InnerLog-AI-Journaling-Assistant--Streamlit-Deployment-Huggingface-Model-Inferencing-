# from ollama import chat

# response = chat(
#     model="gemma3:4b",
#     messages=[
#         {
#             "role": "user",
#             "content": "Give a short, supportive reflection about feeling stressed at work."
#         }
#     ]
# )

# print(response.message.content)



from reflection_agent import ReflectionAgent

# Fictional journal entry for testing
entry = {
    "id": 999,
    "title": "Work Stress",
    "content": (
        "I felt overwhelmed by work today. "
        "I am worried about tomorrow's presentation, "
        "but talking to my friend helped me feel calmer."
    ),
    "analysis": {
        "sentiment": "mixed",
        "emotions": ["anxiety", "relief"],
        "explanation": (
            "The writer experiences work-related anxiety "
            "but also relief after receiving support."
        )
    }
}

try:
    agent = ReflectionAgent()

    print("Starting Reflection Agent...")
    result = agent.reflect(entry, history=[])

    print("\nReflection:")
    print(result.reflection)

    print("\nQuestion:")
    print(result.question)

except Exception as exc:
    import traceback
    traceback.print_exc()

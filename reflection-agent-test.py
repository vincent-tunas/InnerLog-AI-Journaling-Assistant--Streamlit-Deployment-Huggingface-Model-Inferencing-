from ollama import chat

response = chat(
    model="gemma3:4b",
    messages=[
        {
            "role": "user",
            "content": "Give a short, supportive reflection about feeling stressed at work."
        }
    ]
)

print(response.message.content)
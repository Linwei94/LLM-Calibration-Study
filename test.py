from together import Together
import os

# Set the API key for Together
os.environ["TOGETHER_API_KEY"] = "b3bbd1c4faba28bb4b8f910e3c1d15715957533aa72d40f3d23dd5fdc059e7d8"

client = Together()

response = client.chat.completions.create(
    model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    messages=[{"role": "user", "content": "What are some fun things to do in New York?"}],
    logprobs=1,
    max_tokens=10,
)

print(response.choices[0].message.content)
print(response.choices[0].logprobs.token_logprobs)
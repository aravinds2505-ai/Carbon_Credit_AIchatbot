
from openai import OpenAI
import os

GROQ_API_KEY = "REDACTED_GROQ_API_KEY"
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

response = client.responses.create(
    input="Who is the current president of the United States?",
    model="openai/gpt-oss-20b",
)
print(response.output_text)
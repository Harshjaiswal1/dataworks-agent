# llm.py
import os
import requests

AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN environment variable not set.")

def call_llm(prompt: str) -> str:
    url = "https://api.ai-proxy.example/llm"  # Replace with your actual endpoint.
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o-mini",
        "prompt": prompt,
        "max_tokens": 100
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception("LLM API error: " + response.text)
    result = response.json().get("result", "")
    return result

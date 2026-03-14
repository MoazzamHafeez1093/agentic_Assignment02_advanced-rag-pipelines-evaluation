import os
import yaml
import time
import re
from groq import Groq

def get_groq_client():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    api_key = config.get("groq_api_key")
    if not api_key:
        raise ValueError("groq_api_key not found in config.yaml")
    return Groq(api_key=api_key)

def get_model_name():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("generation_model", "llama-3.3-70b-versatile")

def _call_llm_with_retry(client, model, messages, response_format):
    max_retries = 20 # High enough to ride out rolling daily token limits
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=500,
                response_format=response_format
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            error_msg = str(e)
            # Look for Groq's rate limit window: "Please try again in 24m43.488s."
            if "Rate limit reached" in error_msg and "Please try again in" in error_msg:
                match = re.search(r"Please try again in (?:(\d+)h)?(?:(\d+)m)?([\d\.]+)s", error_msg)
                if match:
                    hours = float(match.group(1)) if match.group(1) else 0.0
                    mins = float(match.group(2)) if match.group(2) else 0.0
                    secs = float(match.group(3)) if match.group(3) else 0.0
                    wait_time = hours * 3600 + mins * 60 + secs + 2.0 # 2 seconds buffer
                    print(f"    [Rate Limit] Waiting {wait_time:.1f} seconds for token bucket to refill...")
                    time.sleep(wait_time)
                    continue
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"    [API Error] {error_msg}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e

def call_llm(prompt: str, system_message: str = "You are a helpful and accurate assistant.", json_mode: bool = False) -> str:
    client = get_groq_client()
    model = get_model_name()
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    
    response_format = {"type": "json_object"} if json_mode else None
    
    try:
        return _call_llm_with_retry(client, model, messages, response_format)
    except Exception as e:
        print(f"Error calling LLM after retries: {e}")
        # Add a sleep to prevent immediate failure cascade
        time.sleep(5)
        return ""

def format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant context found."
    context_str = ""
    for i, c in enumerate(chunks):
        context_str += f"--- Source [{i+1}] (From: {c.get('page_url', 'Unknown')}) ---\n"
        context_str += c.get('text', '') + "\n\n"
    return context_str

def generate(query: str, context_chunks: list[dict]) -> str:
    """
    Generate an answer using the provided retrieved context.
    """
    context_str = format_context(context_chunks)
    
    prompt = f"""You are a smart assistant that answers factual questions accurately.
Use the provided retrieved context to answer the user's question.
If the context does not contain enough information to answer the question, just answer saying that you do not know based on the context. Keep the answer concise and direct.

Context:
{context_str}

Question: {query}
Answer:"""

    return call_llm(prompt)

def generate_queries(query: str, n: int = 3) -> list[str]:
    """
    Generate multiple variant phrasings of the input query (for RAG Fusion).
    """
    prompt = f"""Generate {n} different variations of the following search query to help retrieve more comprehensive results. 
Return ONLY the queries, one per line, with no numbered lists or extra text.

Original query: {query}"""
    
    result = call_llm(prompt)
    queries = [q.strip("- 1234567890. ") for q in result.split("\n") if q.strip()]
    return queries[:n]

def generate_hypothetical_doc(query: str) -> str:
    """
    Generate a hypothetical document/answer to the query (for HyDE).
    """
    prompt = f"""Please write a 1 to 2 paragraph hypothetical document that directly answers the following question. Provide specific, detailed (even if plausible but fictional) facts that might appear in a real document answering this. Do not mention that this is hypothetical.

Question: {query}"""
    
    return call_llm(prompt)

def assess_confidence(query: str, context_chunks: list[dict]) -> str:
    """
    Assess whether the retrieved context contains the answer to the query (for CRAG).
    Returns "high" or "low".
    """
    context_str = format_context(context_chunks)
    prompt = f"""Evaluate whether the following retrieved context passages contain sufficient information to answer the question.
If the context directly answers or contains enough facts to answer the question, output exactly: {{"confidence": "high"}}
If the context is irrelevant or insufficient, output exactly: {{"confidence": "low"}}

Context:
{context_str}

Question: {query}"""
    
    result = call_llm(prompt, system_message="Return JSON.", json_mode=True)
    if "high" in result.lower():
        return "high"
    return "low"

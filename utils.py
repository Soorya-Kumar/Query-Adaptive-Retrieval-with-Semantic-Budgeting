import os
import time
import httpx

def color_print(text):
    """Print in bright neon green"""
    print(f"\033[1;92m{text}\033[0m")

def post_with_retries(url, json, retries=3, delay=3):
    for attempt in range(retries):
        try:
            response = httpx.post(url, json=json, timeout=120)
            response.raise_for_status()
            return response
        except httpx.RequestError as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise
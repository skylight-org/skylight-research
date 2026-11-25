#!/usr/bin/env python3
"""
Simple API-based Chat Client for Sparse Attention Models.
Connects to the start_server.py OpenAI-compatible endpoint.
"""

import argparse
import sys
import json
import requests

def get_multiline_input(label):
    """Gets multiline input from the user until 'EOF' is typed."""
    print(f"\n$ {label} =")
    print("(Enter text. Type 'EOF' on a new line to finish, or 'exit' to quit)")
    lines = []
    while True:
        try:
            line = input()
            if line.strip().lower() == "exit":
                sys.exit(0)
            if line.strip().upper() == "EOF":
                break
            lines.append(line)
        except (EOFError, KeyboardInterrupt):
            return None
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="API Chat Client for Sparse Attention Models")
    parser.add_argument("--port", type=int, default=4000, help="Port number for the API server (default: 4000)")
    args = parser.parse_args()
    
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    print(f"--- API Chat Client ---")
    print(f"Connecting to: {url}")
    
    messages = []

    while True:
        context = get_multiline_input("context")
        if context is None: break
        
        question = get_multiline_input("question")
        if question is None: break

        if not context.strip() and not question.strip():
            continue

        # Combine for the prompt
        user_content = f"Context:\n{context}\n\nQuestion:\n{question}"
        messages.append({"role": "user", "content": user_content})

        print("\nGenerating response...")
        try:
            response = requests.post(
                url,
                json={
                    "model": "sparse-model",
                    "messages": messages,
                    "max_tokens": 1024,
                    "temperature": 0.7
                },
                timeout=300
            )
            response.raise_for_status()
            
            data = response.json()
            answer = data['choices'][0]['message']['content']
            
            print("\n$ <Response>:")
            print(answer)
            
            # Add assistant response to history
            messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            print(f"\n[API ERROR] {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Details: {e.response.text}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python

import os
import argparse
import json
import time
import re
import numpy as np
import concurrent.futures
from datasets import load_dataset
from openai import OpenAI

def debug_print_code_structure(code_snippet):
    """Prints the structure of the code snippet (for debugging purposes)."""
    print("Code snippet structure:")
    print("Type:", type(code_snippet))
    lines = code_snippet.splitlines()
    print("Number of lines:", len(lines))
    for i, line in enumerate(lines):
        print(f"Line {i}: {line[:80]}")

def sample_severity():
    """
    Sample a noise level from a Gaussian distribution (clipped between 0 and 1)
    and map it to a textual description of the bug severity.
    """
    noise = np.clip(np.random.normal(loc=0.5, scale=0.2), 0, 1)
    if noise < 0.33:
        return "minor syntax errors", "typo"
    elif noise < 0.66:
        return "moderate logical and syntax errors", "novice programmer"
    else:
        return "severe and profound bugs resulting in broken_code", "malfunctioning AI"

def parse_modified_code(response_text):
    """
    Extracts the code between <<<START_CODE>>> and <<<END_CODE>>> tags.
    """
    pattern = re.compile(r"<<<START_CODE>>>(.*?)<<<END_CODE>>>", re.DOTALL)
    match = pattern.search(response_text)
    if match:
        return match.group(1).strip()
    else:
        print("Warning: Could not find <<<START_CODE>>> tags in the response.")
        return response_text.strip()

def parse_explanation(response_text):
    """
    Extracts the explanation between <<<START_EXPLANATION>>> and <<<END_EXPLANATION>>> tags.
    """
    pattern = re.compile(r"<<<START_EXPLANATION>>>(.*?)<<<END_EXPLANATION>>>", re.DOTALL)
    match = pattern.search(response_text)
    if match:
        return match.group(1).strip()
    else:
        print("Warning: Could not find <<<START_EXPLANATION>>> tags in the response.")
        return ""

def inject_bugs(code_snippet, client, model):
    """
    Constructs a prompt to inject bugs into the given code snippet and calls the API.
    The prompt instructs the model to return only the modified code wrapped in
    <<<START_CODE>>> and <<<END_CODE>>> tags and a brief explanation in
    <<<START_EXPLANATION>>> and <<<END_EXPLANATION>>> tags, with no additional commentary.
    
    Uses an exponential backoff strategy for rate-limit errors.
    
    If the response does not contain the required tags, the function sends one follow-up
    message (continuing the chat) requesting a strictly formatted answer. If that follow-up
    still fails, the conversation is restarted.
    """
    severity, persona = sample_severity()
    original_prompt = (
        f"Inject bugs into the following Python code so that the resulting code is exactly the same format as the input but with injected bugs. "
        f"The bugs should be {severity} like a {persona} would introduce. "
        f"The modified code must be returned inside the tags <<<START_CODE>>> and <<<END_CODE>>> with no additional commentary, explanations, or comments.\n\n"
        f"Do NOT include any comments on what is wrong with the code. The model should only return the modified code. "
        f"Ensure that you output the entire code snippet, including any indentation and newlines.\n\n"
        f"Do not stub, remove, or modify any part of the code. The code should be returned exactly as it was input, only with bugs added.\n\n"
        f"Then, briefly state the bugs that you injected into the code after the closing <<<END_CODE>>> tag in its own "
        f"<<<START_EXPLANATION>>> and <<<END_EXPLANATION>>> tags.\n\n"
        f"For example, your response should look like this:\n\n"
        f"<<<START_CODE>>>\n"
        f"def foo();\n"
        f"    print('Hello, world!')\n"
        f"<<<END_CODE>>>\n"
        f"<<<START_EXPLANATION>>>\n"
        f"Used a semicolon instead of a colon after def foo().\n"
        f"<<<END_EXPLANATION>>>\n\n"
        f"Be precise in the formatting of your response. Do not include any additional text or comments. "
        f"Any deviation from the expected format may result in a rejection of the response.\n\n"
        f"Code: \n\n{code_snippet}"
    )
    
    max_retries = 5
    attempt = 0
    backoff = 2  # seconds
    
    while attempt < max_retries:
        # Start a new conversation context for each attempt.
        messages = [{"role": "user", "content": original_prompt}]
        followup_attempted = False
        try:
            while True:
                try:
                    completion = client.chat.completions.create(
                        extra_headers={"X-Title": "fuzzy"},
                        extra_body={},
                        model=model,
                        messages=messages
                    )
                    if getattr(completion, "error", None) is not None:
                        error_message = completion.error.get("message", "")
                        if "Rate limit exceeded" in error_message:
                            raise Exception("Rate limit exceeded")
                    if not completion or not getattr(completion, "choices", None):
                        print("Unexpected completion structure:", completion)
                        return None, None
                    response_text = completion.choices[0].message.content

                    # Check for required tags.
                    if (("<<<START_CODE>>>" in response_text) and ("<<<END_CODE>>>" in response_text)
                        and ("<<<START_EXPLANATION>>>" in response_text) and ("<<<END_EXPLANATION>>>" in response_text)):
                        return parse_modified_code(response_text), parse_explanation(response_text)
                    else:
                        if not followup_attempted:
                            # Continue the chat once with a follow-up.
                            followup = (
                                "Your previous response did not follow the expected format. "
                                "Please reformat your answer so that it strictly includes only the modified code "
                                "wrapped in <<<START_CODE>>> and <<<END_CODE>>> tags and the explanation "
                                "wrapped in <<<START_EXPLANATION>>> and <<<END_EXPLANATION>>> tags, with no additional commentary."
                            )
                            messages.append({"role": "assistant", "content": response_text})
                            messages.append({"role": "user", "content": followup})
                            followup_attempted = True
                            print("Incorrect format detected; continuing the chat for reformatting.")
                            continue  # Re-send within the same context.
                        else:
                            # Already attempted follow-up; break out to retry from scratch.
                            print("Follow-up did not yield the expected format. Restarting conversation.")
                            break
                except Exception as e:
                    if "Rate limit exceeded" in str(e):
                        raise  # Re-raise to be caught by the outer try-except
                    else:
                        print("API request failed with exception:", e)
                        debug_print_code_structure(code_snippet)
                        return None, None
        except Exception as e:
            if "Rate limit exceeded" in str(e):
                attempt += 1
                print(f"Rate limit error encountered (attempt {attempt}/{max_retries}). Backing off for {backoff} seconds.")
                time.sleep(backoff)
                backoff *= 2
                continue
        
        # If we reach here, it means the inner loop broke out normally (not via exception)
        attempt += 1
    
    print("Max retries exceeded for this code snippet.")
    return None, None

def main():
    parser = argparse.ArgumentParser(
        description="Fuzzy: Spawn API requests in groups to inject bugs into Python code from the CodeParrot dataset."
    )
    parser.add_argument("--group_size", type=int, default=100, help="Number of concurrent requests to spawn at once")
    parser.add_argument("--max_requests", type=int, default=None, help="Optional: Maximum number of samples to process")
    parser.add_argument("--model", type=str, default="deepseek/deepseek-chat:free", help="Model to use for bug injection")
    parser.add_argument("--output_file", type=str, default="bug_injected_results.jsonl", help="Output file (JSON Lines)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Environment variable OPENROUTER_API_KEY is not set.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Stream the dataset from Hugging Face.
    print("Loading dataset (streaming from Hugging Face)...")
    dataset = load_dataset("codeparrot/codeparrot-clean-train", split="train", streaming=True)
    
    # Iterate over individual samples (each sample should have a "content" field).
    sample_iter = iter(dataset)
    
    processed_count = 0
    group_count = 0

    with open(args.output_file, "a") as out_file, \
         concurrent.futures.ThreadPoolExecutor(max_workers=args.group_size) as executor:
        while True:
            futures = []
            # Collect a group of requests.
            for _ in range(args.group_size):
                try:
                    sample = next(sample_iter)
                except StopIteration:
                    break
                original_code = sample.get("content")
                future = executor.submit(inject_bugs, original_code, client, args.model)
                futures.append((future, original_code))
                processed_count += 1
                if args.max_requests is not None and processed_count >= args.max_requests:
                    break

            if not futures:
                break  # No more samples.
            
            group_count += 1
            group_start_time = time.time()
            print(f"Processing group {group_count} with {len(futures)} requests...")
            for future, original_code in futures:
                try:
                    modified_code, explanation = future.result()
                    if modified_code is not None:
                        result = {
                            "original_code": original_code,
                            "modified_code": modified_code,
                            "explanation": explanation
                        }
                        out_file.write(json.dumps(result) + "\n")
                except Exception as e:
                    print("Error processing a sample:", e)
            out_file.flush()
            group_duration = time.time() - group_start_time
            print(f"Group {group_count} processed in {group_duration:.2f} seconds.")
            
            if args.max_requests is not None and processed_count >= args.max_requests:
                break

    print(f"Completed processing {processed_count} samples. Results appended to {args.output_file}.")

if __name__ == "__main__":
    main()

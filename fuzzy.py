#!/usr/bin/env python

import os
import argparse
import json
import time
import numpy as np
import pandas as pd
import concurrent.futures

# Import the OpenAI client (this uses the provided boilerplate style)
from openai import OpenAI

def sample_severity():
    """
    Sample a noise level from a Gaussian distribution (clipped between 0 and 1)
    and map it to a textual description of the bug severity.
    """
    noise = np.clip(np.random.normal(loc=0.5, scale=0.2), 0, 1)
    if noise < 0.33:
        return "minor syntax errors"
    elif noise < 0.66:
        return "moderate logical and syntax errors"
    else:
        return "severe and profound bugs resulting in broken code"

def inject_bugs(code_snippet, client):
    """
    Given a code snippet and a pre-initialized client, construct a prompt that
    instructs the model to inject bugs. Returns the modified code.
    """
    severity = sample_severity()
    prompt = (
        f"Given the following Python code, inject bugs intentionally to simulate Gaussian noise. "
        f"The errors should be {severity}. Here is the code:\n\n{code_snippet}\n\n"
        "Return the modified code with injected errors."
    )

    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://fuzzy.app",  # Replace with your site URL if desired.
                "X-Title": "fuzzy",  # Name of the app.
            },
            extra_body={},
            model="deepseek/deepseek-chat:free",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"API request failed: {e}")
        return None

def process_batch(batch_df, client):
    """
    Process one batch of code snippets from the dataframe by sending each one
    to the API endpoint to inject bugs.
    """
    results = []
    for idx, row in batch_df.iterrows():
        code = row["content"]
        modified_code = inject_bugs(code, client)
        if modified_code is not None:
            results.append({
                "original_code": code,
                "modified_code": modified_code,
            })
        # Optional sleep to avoid rate limits.
        time.sleep(0.1)
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Fuzzy: Inject bugs into Python code from the CodeParrot dataset using an OpenAI-compatible API."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to the Parquet dataset file")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of queries per batch")
    parser.add_argument("--max_workers", type=int, default=10, help="Number of concurrent threads")
    args = parser.parse_args()

    # Retrieve the API key from the environment
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Environment variable OPENROUTER_API_KEY is not set.")

    # Initialize the OpenAI client using the OpenRouter API endpoint.
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    print("Loading dataset...")
    df = pd.read_parquet(args.dataset)
    total_samples = len(df)
    print(f"Loaded dataset with {total_samples} code samples.")

    # Split the dataframe into batches.
    batches = [df[i:i + args.batch_size] for i in range(0, total_samples, args.batch_size)]
    print(f"Processing {len(batches)} batches (batch size: {args.batch_size}).")

    all_results = []
    # Use ThreadPoolExecutor to parallelize batch processing.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_batch = {
            executor.submit(process_batch, batch, client): batch
            for batch in batches
        }
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_result = future.result()
            all_results.extend(batch_result)

    # Save the results to a JSON file.
    output_file = "bug_injected_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Completed processing. Results saved to {output_file}.")

if __name__ == "__main__":
    main()


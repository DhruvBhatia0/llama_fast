"""
run: python3 -m torch.distributed.run --nproc_per_node=1 ../profile.py 

run this out of the llama repo
"""

import sys
from pathlib import Path

# Add the parent directory to sys.path to import the llama module
sys.path.append(str(Path(__file__).parent.parent))

from llama import Dialog, Llama

# Path to the Llama 3 8B weights
WEIGHTS_PATH = "/home/dhruv/llama_fast/llama3/Meta-Llama-3-8B"

def main():
    # Initialize the Llama model
    generator = Llama.build(
        ckpt_dir=WEIGHTS_PATH,
        tokenizer_path=f"{WEIGHTS_PATH}/tokenizer.model",
        max_seq_len=512,
        max_batch_size=1,
    )

    # Sample input text
    input_text = "Explain the concept of artificial intelligence in simple terms."

    # Prepare the input for the model
    dialog = [
        {"role": "user", "content": input_text}
    ]

    # Generate a response
    response = generator.chat_completion(
        [dialog],  # Wrap the dialog in a list as the function expects a list of dialogs
        max_gen_len=256,
        temperature=0.7,
        top_p=0.9,
    )

    # Extract and print the model's response
    output = response[0]['generation']['content']
    
    print("Input:", input_text)
    print("\nOutput:", output)

if __name__ == "__main__":
    main()
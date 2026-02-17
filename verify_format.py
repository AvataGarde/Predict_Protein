
import sys
import os
import re

# Add path to import functions
sys.path.append("/users/thz501/data/bio")

from sft import format_instruction_veupathdb
from evaluate import strip_model_prefix

class MockTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return messages[-1]["content"]

def test_sft_format():
    print("Testing sft.py format_instruction_veupathdb...")
    sample = {
        "Gene_ID": "TEST_001",
        "user_prompt": "This is a gene summary.",
        "PRODUCT_NAME": "Test Protein"
    }
    tokenizer = MockTokenizer()
    result = format_instruction_veupathdb(sample, tokenizer)
    
    # Check if assistant content has correct prefix
    assistant_content = result["text"]
    expected_prefix = "Product_Description: "
    if assistant_content.startswith(expected_prefix):
        print(f"PASS: sft.py prompt format correct. starts with '{expected_prefix}'")
    else:
        print(f"FAIL: sft.py prompt format incorrect. Got: '{assistant_content}'")

def test_evaluate_strip():
    print("\nTesting evaluate.py strip_model_prefix...")
    
    test_cases = [
        ("Product_Description: Test Protein", "Test Protein"),
        ("Protein_Description: Test Protein", "Test Protein"), # Legacy support
        ("Product Description: Test Protein", "Test Protein"),
        ("Protein Name: Test Protein", "Test Protein"),
        ("Test Protein", "Test Protein"),
         ("Product_Description:   Test Protein  ", "Test Protein"),
    ]
    
    all_pass = True
    for input_str, expected in test_cases:
        result = strip_model_prefix(input_str)
        if result == expected:
            print(f"PASS: '{input_str}' -> '{result}'")
        else:
            print(f"FAIL: '{input_str}' -> '{result}' (Expected: '{expected}')")
            all_pass = False
            
    if all_pass:
        print("All evaluate.py strip tests passed.")

if __name__ == "__main__":
    try:
        test_sft_format()
        test_evaluate_strip()
    except Exception as e:
        print(f"An error occurred: {e}")

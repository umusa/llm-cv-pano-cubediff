#!/usr/bin/env python3
"""
Simple script to convert a text file containing Jupyter notebook JSON to a valid .ipynb file.
This solves the issue of the notebook being corrupted or improperly formatted when downloaded.
"""

import json
import sys
import os

def convert_txt_to_ipynb(input_file, output_file=None):
    """
    Convert a text file containing notebook JSON to a valid .ipynb file
    
    Args:
        input_file: Path to input text file
        output_file: Path to output .ipynb file (optional)
    
    Returns:
        Path to the created .ipynb file
    """
    # Determine output file name if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.ipynb"
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        notebook_json = f.read()
    
    # Parse as JSON
    try:
        notebook_data = json.loads(notebook_json)
    except json.JSONDecodeError as e:
        print(f"Error parsing notebook JSON: {e}")
        return None
    
    # Write formatted JSON to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook_data, f, indent=1)
    
    print(f"Successfully converted {input_file} to {output_file}")
    return output_file

def main():
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python txt_to_ipynb_converter.py input_file.txt [output_file.ipynb]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Convert the file
    result = convert_txt_to_ipynb(input_file, output_file)
    
    if result is None:
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Robust script to convert a text file containing Jupyter notebook JSON to a valid .ipynb file.
This handles potential JSON parsing errors by cleaning the input file.
"""

import json
import sys
import os
import re

def clean_json_text(text):
    """Clean JSON text to handle common formatting issues"""
    # Fix common JSON issues
    
    # 1. Remove any trailing commas in arrays and objects
    text = re.sub(r',(\s*[\]}])', r'\1', text)
    
    # 2. Make sure all strings have proper double quotes
    # This is more complex and might need manual intervention
    
    # 3. Remove any lines that might contain extra data outside the JSON structure
    try:
        # First attempt: try to parse as is
        json.loads(text)
        return text
    except json.JSONDecodeError as e:
        error_line = e.lineno
        error_col = e.colno
        error_msg = str(e)
        print(f"JSON error at line {error_line}, column {error_col}: {error_msg}")
        
        # Get all lines
        lines = text.splitlines()
        
        # Attempt to fix the problematic line
        if error_line <= len(lines):
            problematic_line = lines[error_line - 1]
            print(f"Problematic line: {problematic_line[:50]}...")
            
            # Try to fix common issues
            if "Extra data" in error_msg:
                # If there's extra data after JSON, truncate the file at the error point
                return '\n'.join(lines[:error_line - 1])
            
            if "Expecting" in error_msg:
                # For syntax errors, attempt simple fixes
                if "property name" in error_msg:
                    # Missing quotes around property names
                    fixed_line = re.sub(r'(\s*)(\w+)(\s*:)', r'\1"\2"\3', problematic_line)
                    lines[error_line - 1] = fixed_line
                
                # More fixes can be added here for common errors
        
        # Return the potentially fixed text
        return '\n'.join(lines)

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
    
    # Clean the JSON text
    cleaned_json = clean_json_text(notebook_json)
    
    # Parse as JSON
    try:
        notebook_data = json.loads(cleaned_json)
    except json.JSONDecodeError as e:
        print(f"Still error parsing notebook JSON after cleaning: {e}")
        
        # As a fallback, let's try the manual approach
        print("Trying manual rebuilding of the notebook structure...")
        notebook_data = create_basic_notebook_structure(input_file)
        
        if notebook_data is None:
            return None
    
    # Write formatted JSON to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook_data, f, indent=1)
    
    print(f"Successfully converted {input_file} to {output_file}")
    return output_file

def create_basic_notebook_structure(input_file):
    """Create a basic notebook structure by parsing cells manually"""
    cells = []
    current_cell_type = None
    current_cell_content = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Look for patterns that indicate cells
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check for cell markers
        if '"cell_type": "markdown"' in line:
            # If we were already building a cell, add it to cells
            if current_cell_type is not None:
                cells.append({
                    "cell_type": current_cell_type,
                    "source": current_cell_content,
                    "metadata": {},
                    "outputs": [] if current_cell_type == "code" else None
                })
            
            # Start a new markdown cell
            current_cell_type = "markdown"
            current_cell_content = []
            
            # Skip ahead to the "source" part
            while i < len(lines) and '"source": [' not in lines[i]:
                i += 1
            
            # Collect the source content
            i += 1  # Skip the line with "source": [
            while i < len(lines) and ']' not in lines[i]:
                source_line = lines[i].strip()
                if source_line.startswith('"') and source_line.endswith('",'):
                    source_line = source_line[1:-2]  # Remove quotes and comma
                elif source_line.startswith('"') and source_line.endswith('"'):
                    source_line = source_line[1:-1]  # Remove just quotes
                
                # Unescape any escaped quotes
                source_line = source_line.replace('\\"', '"')
                
                current_cell_content.append(source_line + '\n')
                i += 1
        
        elif '"cell_type": "code"' in line:
            # If we were already building a cell, add it to cells
            if current_cell_type is not None:
                cells.append({
                    "cell_type": current_cell_type,
                    "source": current_cell_content,
                    "metadata": {},
                    "outputs": [] if current_cell_type == "code" else None
                })
            
            # Start a new code cell
            current_cell_type = "code"
            current_cell_content = []
            
            # Skip ahead to the "source" part
            while i < len(lines) and '"source": [' not in lines[i]:
                i += 1
            
            # Collect the source content
            i += 1  # Skip the line with "source": [
            while i < len(lines) and ']' not in lines[i]:
                source_line = lines[i].strip()
                if source_line.startswith('"') and source_line.endswith('",'):
                    source_line = source_line[1:-2]  # Remove quotes and comma
                elif source_line.startswith('"') and source_line.endswith('"'):
                    source_line = source_line[1:-1]  # Remove just quotes
                
                # Unescape any escaped quotes
                source_line = source_line.replace('\\"', '"')
                
                current_cell_content.append(source_line + '\n')
                i += 1
        
        i += 1
    
    # Add the last cell if there is one
    if current_cell_type is not None:
        cells.append({
            "cell_type": current_cell_type,
            "source": current_cell_content,
            "metadata": {},
            "outputs": [] if current_cell_type == "code" else None
        })
    
    # Create the notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook

def main():
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python robust_txt_to_ipynb.py input_file.txt [output_file.ipynb]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Convert the file
    result = convert_txt_to_ipynb(input_file, output_file)
    
    if result is None:
        sys.exit(1)

if __name__ == "__main__":
    main()

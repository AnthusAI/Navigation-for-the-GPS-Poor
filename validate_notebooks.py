#!/usr/bin/env python
"""
Validate all Jupyter notebooks in the project for syntax errors.

This script checks all .ipynb files for Python syntax errors without
needing to execute them. Useful for catching issues before running.

Usage:
    python validate_notebooks.py                    # Check all notebooks
    python validate_notebooks.py chapters/1         # Check specific directory
    python validate_notebooks.py chapters/1/demo.ipynb  # Check specific notebook
"""

import json
import ast
import sys
from pathlib import Path
from typing import List, Tuple


def validate_notebook(notebook_path: Path) -> Tuple[bool, List[Tuple[int, str]]]:
    """
    Validate a single notebook for syntax errors.
    
    Args:
        notebook_path: Path to the .ipynb file
        
    Returns:
        (is_valid, errors) where errors is a list of (cell_num, error_msg) tuples
    """
    try:
        with open(notebook_path, 'r') as f:
            nb = json.load(f)
    except Exception as e:
        return False, [(0, f"Failed to load notebook: {e}")]
    
    errors = []
    code_cells = [(i, cell) for i, cell in enumerate(nb['cells']) if cell['cell_type'] == 'code']
    
    for cell_num, cell in code_cells:
        source = cell['source']
        # Handle both string and list formats
        if isinstance(source, list):
            source = ''.join(source)
        
        # Filter out IPython magic commands and shell commands
        # These are valid in notebooks but not in pure Python
        filtered_lines = []
        for line in source.split('\n'):
            stripped = line.lstrip()
            # Skip IPython magics (%, %%), shell commands (!), and help (?)
            if stripped.startswith(('%', '!', '?')):
                continue
            filtered_lines.append(line)
        
        filtered_source = '\n'.join(filtered_lines)
        
        # Skip empty cells
        if not filtered_source.strip():
            continue
        
        try:
            ast.parse(filtered_source)
        except SyntaxError as e:
            errors.append((cell_num, f"Line {e.lineno}: {e.msg}"))
        except Exception as e:
            errors.append((cell_num, f"Parse error: {str(e)}"))
    
    return len(errors) == 0, errors


def find_notebooks(path: Path = None) -> List[Path]:
    """Find all notebooks in the given path or the entire project."""
    if path is None:
        path = Path.cwd()
    
    if path.is_file() and path.suffix == '.ipynb':
        return [path]
    
    # Find all .ipynb files, excluding hidden directories and checkpoints
    notebooks = []
    for nb_path in path.rglob('*.ipynb'):
        # Skip hidden directories and checkpoint directories
        if any(part.startswith('.') for part in nb_path.parts):
            continue
        if '.ipynb_checkpoints' in str(nb_path):
            continue
        notebooks.append(nb_path)
    
    return sorted(notebooks)


def main():
    """Main validation function."""
    # Determine what to check
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        target = Path.cwd()
    
    notebooks = find_notebooks(target)
    
    if not notebooks:
        print("❌ No notebooks found!")
        return 1
    
    print("=" * 80)
    print(f"VALIDATING {len(notebooks)} NOTEBOOK(S)")
    print("=" * 80)
    print()
    
    all_valid = True
    results = []
    
    for nb_path in notebooks:
        rel_path = nb_path.relative_to(Path.cwd()) if nb_path.is_relative_to(Path.cwd()) else nb_path
        
        is_valid, errors = validate_notebook(nb_path)
        results.append((rel_path, is_valid, errors))
        
        if is_valid:
            print(f"✅ {rel_path}")
        else:
            print(f"❌ {rel_path}")
            all_valid = False
            for cell_num, error_msg in errors:
                print(f"   Cell {cell_num}: {error_msg}")
            print()
    
    print()
    print("=" * 80)
    
    if all_valid:
        print(f"🎉 ALL {len(notebooks)} NOTEBOOK(S) ARE VALID!")
        print("=" * 80)
        return 0
    else:
        failed = sum(1 for _, valid, _ in results if not valid)
        print(f"❌ {failed}/{len(notebooks)} NOTEBOOK(S) HAVE ERRORS")
        print("=" * 80)
        return 1


if __name__ == '__main__':
    sys.exit(main())


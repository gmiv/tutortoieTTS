#!/usr/bin/env python3
"""
Script to convert multi-line strings to one-liners.
Reads from clipboard, squeezes to one line, and copies back to clipboard.
"""

import subprocess
import sys

def get_clipboard():
    """Get text from clipboard (WSL-compatible)"""
    try:
        # Try using Windows clipboard from WSL
        result = subprocess.run(['powershell.exe', '-command', 'Get-Clipboard'],
                              capture_output=True, text=True, check=True)
        return result.stdout.rstrip('\n')
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # Fallback to xclip if available
            result = subprocess.run(['xclip', '-selection', 'clipboard', '-o'],
                                  capture_output=True, text=True, check=True)
            return result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: Could not access clipboard. Install xclip or run from WSL with PowerShell access.")
            sys.exit(1)

def set_clipboard(text):
    """Set clipboard text (WSL-compatible)"""
    try:
        # Try using Windows clipboard from WSL
        subprocess.run(['powershell.exe', '-command', f'Set-Clipboard -Value "{text}"'],
                      check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # Fallback to xclip if available
            subprocess.run(['xclip', '-selection', 'clipboard'],
                          input=text.encode(), check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: Could not access clipboard. Install xclip or run from WSL with PowerShell access.")
            sys.exit(1)

def squeeze_to_oneliner(text):
    """Convert multi-line string to a single line and remove special characters"""
    # Replace newlines with spaces
    oneliner = text.replace('\n', ' ').replace('\r', '')

    # Remove or replace special characters that could cause issues in shell
    special_chars = {
        '"': '',      # Remove double quotes
        "'": '',      # Remove single quotes
        '`': '',      # Remove backticks
        '$': '',      # Remove dollar signs
        '\\': '',     # Remove backslashes
        ';': '',      # Remove semicolons
        '&': 'and',   # Replace ampersands
        '|': '',      # Remove pipes
        '(': '',      # Remove parentheses
        ')': '',
        '<': '',      # Remove angle brackets
        '>': '',
        '*': '',      # Remove asterisks
        '?': '',      # Remove question marks
        '!': '',      # Remove exclamation marks
        '\t': ' ',    # Replace tabs with spaces
    }

    for char, replacement in special_chars.items():
        oneliner = oneliner.replace(char, replacement)

    # Compress multiple spaces to single space
    while '  ' in oneliner:
        oneliner = oneliner.replace('  ', ' ')

    return oneliner.strip()

if __name__ == '__main__':
    print("Reading from clipboard...")
    original = get_clipboard()

    if not original:
        print("Clipboard is empty!")
        sys.exit(1)

    print(f"Original length: {len(original)} characters")
    print(f"Original lines: {original.count(chr(10)) + 1}")

    # Convert to one-liner
    oneliner = squeeze_to_oneliner(original)

    print(f"One-liner length: {len(oneliner)} characters")

    # Copy back to clipboard
    set_clipboard(oneliner)
    print("✓ One-liner copied to clipboard!")

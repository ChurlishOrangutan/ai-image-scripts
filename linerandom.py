import random
import argparse

def randomize_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    random.shuffle(lines)

    with open(file_path, 'w') as file:
        file.writelines(lines)

def check_unclosed_parens(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    unclosed_parens_lines = []
    for i, line in enumerate(lines):
        open_parens = line.count('(')
        close_parens = line.count(')')
        if open_parens != close_parens:
            unclosed_parens_lines.append((i + 1, line.strip()))

    if unclosed_parens_lines:
        print("Lines with unclosed parentheses:")
        for line_num, content in unclosed_parens_lines:
            print(f"Line {line_num}: {content}")
    else:
        print("No unclosed parentheses found.")

def main(file_path):
    print(f"Randomizing lines in file: {file_path}")
    randomize_lines(file_path)
    print(f"Checking for unclosed parentheses in file: {file_path}")
    check_unclosed_parens(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomize lines in a text file and check for unclosed parentheses.")
    parser.add_argument("file_path", type=str, help="Path to the text file")
    args = parser.parse_args()
    main(args.file_path)

import re
import random

def parse_pairs(text):
    """
    Parses the text and extracts all data pairs.
    Each pair must contain:
      - A line starting with "Pair <number>:".
      - "Sentence 1 : "<content>" (content inside double quotes).
      - "Sentence 2 : "<content>" (content inside double quotes).
      - "Similarity Score:" followed by a number.
    
    The regex is set to work across newlines (DOTALL)
    and ignores any extra white space or random lines.
    """
    pattern = re.compile(
        r'Pair\s*\d+\s*:\s*.*?'                # Match "Pair x:" and any characters after (non-greedy)
        r'Sentence\s*1\s*:\s*\"([^\"]+)\".*?'    # Capture Sentence 1 content inside quotes
        r'Sentence\s*2\s*:\s*\"([^\"]+)\".*?'    # Capture Sentence 2 content inside quotes
        r'Similarity Score\s*:\s*([0-9]+\.[0-9]+)',  # Capture the similarity score as a float
        re.DOTALL | re.IGNORECASE
    )
    
    matches = pattern.findall(text)
    pairs = []
    for match in matches:
        sentence1 = match[0].strip()
        sentence2 = match[1].strip()
        score = float(match[2])
        pairs.append((sentence1, sentence2, score))
    return pairs

def write_pairs(filename, pairs):
    """
    Writes the list of pairs to a file.
    Each pair is formatted nicely.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for idx, (sentence1, sentence2, score) in enumerate(pairs, start=1):
            f.write(f"Pair {idx}:\n")
            f.write(f"Sentence 1: \"{sentence1}\"\n")
            f.write(f"Sentence 2: \"{sentence2}\"\n")
            f.write(f"Similarity Score: {score}\n")
            f.write("\n")  # Extra newline for readability

def main():
    input_file = "input.txt"              # Your original document
    training_file = "training_data.txt"     # Output file for training data
    testing_file = "testing_data.txt"       # Output file for testing data
    
    # Read the entire content of the input file
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Parse all pairs from the content
    pairs = parse_pairs(content)
    if not pairs:
        print("No pairs found in the input file. Please check the format.")
        return
    
    # Randomly shuffle the pairs for random splitting
    random.shuffle(pairs)
    
    # Determine the split index for an 80/20 split
    split_index = int(0.8 * len(pairs))
    training_pairs = pairs[:split_index]
    testing_pairs = pairs[split_index:]
    
    # Write out the training and testing pairs to their respective files
    write_pairs(training_file, training_pairs)
    write_pairs(testing_file, testing_pairs)
    
    # Print a summary of the process
    print(f"Total pairs found: {len(pairs)}")
    print(f"Training pairs: {len(training_pairs)} written to '{training_file}'")
    print(f"Testing pairs: {len(testing_pairs)} written to '{testing_file}'")

if __name__ == "__main__":
    main()


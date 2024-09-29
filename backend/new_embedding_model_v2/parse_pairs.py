import re

# Function to load and parse pairs from the file
def load_pairs_from_file(file_path):
    pairs = []
    current_pair = {}

    # Regular expression for the compact format
    compact_format_re = re.compile(r'Pair \d+: Sentence 1: "(.*)" Sentence 2: "(.*)" Similarity Score: ([0-1]\.\d+)')

    with open(file_path, 'r') as file:
        for line in file:
            # Strip any bullet points or unexpected leading characters
            line = line.lstrip('•-–* ').strip()

            # Check for the compact format
            compact_match = compact_format_re.match(line)
            if compact_match:
                sentence_1 = compact_match.group(1)
                sentence_2 = compact_match.group(2)
                score = float(compact_match.group(3))

                pairs.append({
                    'sentence_1': sentence_1,
                    'sentence_2': sentence_2,
                    'score': score
                })
                continue  # Skip to the next line as this pair is complete
            
            # Check for the detailed format
            if line.startswith("Pair"):
                if current_pair:  # Save the last pair before starting a new one
                    pairs.append(current_pair)
                    current_pair = {}

            elif line.startswith("Sentence 1:"):
                current_pair['sentence_1'] = line.split("Sentence 1:", 1)[1].strip().strip('"')

            elif line.startswith("Sentence 2:"):
                current_pair['sentence_2'] = line.split("Sentence 2:", 1)[1].strip().strip('"')

            elif line.startswith("Similarity Score:"):
                current_pair['score'] = float(line.split("Similarity Score:", 1)[1].strip())

        # Add the last pair if it's complete
        if current_pair and 'sentence_1' in current_pair and 'sentence_2' in current_pair and 'score' in current_pair:
            pairs.append(current_pair)
    
    return pairs

import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
import unicodedata
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Helper function to make strings ASCII-safe
def ascii_safe_string(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

# Extract text from the PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
    except Exception as e:
        print(f"An error occurred while extracting text from PDF: {e}")
    return text

# Parse the extracted text into menu items
def parse_menu_text(menu_text):
    menu_items = []
    lines = menu_text.split('\n')
    current_item = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.search(r'(\$[0-9.]+)', line)
        if match:
            price = match.group(1).strip()
            if current_item:
                current_item['price'] = price
                menu_items.append(current_item)
                current_item = None
        else:
            if current_item is None:
                current_item = {'name': line, 'description': '', 'price': ''}
            else:
                current_item['description'] += ' ' + line

    return menu_items

# Vectorize the menu items
def vectorize_items(menu_items, model):
    vectors = []
    for item in menu_items:
        combined_text = f"{item['name']} {item['description']}"
        vector = model.encode(combined_text).tolist()
        
        # Convert name to ASCII-safe format for ID
        safe_id = ascii_safe_string(item['name'])
        if not safe_id:
            # Skip items that cannot be converted to ASCII
            print(f"Skipping item: {item['name']} due to non-ASCII characters.")
            continue

        # Limit ID length and ensure it is unique
        safe_id = safe_id[:50]

        # Sanitize the name and description metadata
        safe_name = ascii_safe_string(item['name'])
        safe_description = ascii_safe_string(item['description'])

        vectors.append({
            "id": safe_id,
            "values": vector,
            "metadata": {
                "name": safe_name,
                "description": safe_description,
                "price": item['price']
            }
        })
    return vectors

# Upload vectors to Pinecone
def upload_vectors_to_pinecone(vectors, index):
    try:
        # Print debug info for vectors
        for vector in vectors:
            print(f"Uploading vector ID: {vector['id']}, Metadata: {vector['metadata']}")

        # Upsert vectors
        index.upsert(vectors=vectors)
        print("Vectors uploaded successfully!")
    except Exception as e:
        print(f"An error occurred during upsert: {e}")

# Main execution
def main():
    # Retrieve Pinecone API key from environment variable
    api_key = os.getenv("PINECONE_API_KEY")
    print(f"Using API key: {api_key}")  # Verify the key
    if not api_key:
        print("Pinecone API key not found in environment variables.")
        return

    # Initialize Pinecone
    pc = Pinecone(api_key=api_key)
    index_name = "unique-food-recommendations"

    # Create or connect to the index
    if index_name not in [index.name for index in pc.list_indexes()]:
        pc.create_index(
            name=index_name, 
            dimension=384,  # Make sure this matches your vector dimensions
            metric='cosine',  # Example metric, can be changed
            spec=ServerlessSpec(
                cloud='aws',
                region="us-east-1"
            )
        )
    index = pc.Index(index_name)

    # Model for vectorization
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Determine the script's directory and set it as the current working directory
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)

    # Relative path to the PDF
    pdf_path = "new_menu.pdf"

    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    # Extract and process text from PDF
    menu_text = extract_text_from_pdf(pdf_path)
    menu_items = parse_menu_text(menu_text)
    vectors = vectorize_items(menu_items, model)

    # Upload vectors to Pinecone
    upload_vectors_to_pinecone(vectors, index)

if __name__ == "__main__":
    main()

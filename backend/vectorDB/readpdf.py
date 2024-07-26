import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, Index, ServerlessSpec, PineconeProtocolError
import os
import unicodedata

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
        vector = model.encode(combined_text)
        
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
            "values": vector.tolist(),
            "metadata": {
                "name": safe_name,
                "description": safe_description,
                "price": item['price']
            }
        })
    return vectors

# Upload vectors to Pinecone with error handling
def upload_vectors_to_pinecone(vectors, index_name, api_key, environment):
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)

        # Ensure the index exists or create it
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

        # Retrieve the host URL for the index
        index_description = pc.describe_index(index_name)
        host = index_description.host

        # Access the index
        index = Index(index_name, host=host)

        # Upsert vectors with error handling for reinitialization
        try:
            index.upsert(vectors)
            print("Vectors uploaded successfully!")
        except PineconeProtocolError:
            print("Encountered a connection issue. Reinitializing Pinecone client.")
            pc = Pinecone(api_key=api_key, environment=environment)
            index = Index(index_name, host=host)
            index.upsert(vectors)
            print("Vectors re-uploaded successfully after reinitialization.")

    except Exception as e:
        print(f"An error occurred while uploading vectors to Pinecone: {e}")

# Main execution
def main():
    # Replace with your Pinecone API key and environment
    api_key = "5e726cf8-5d0d-456c-addf-cc6b5569ea47"  # Replace with your actual API key
    environment = "us-east-1"
    index_name = "unique-food-recommendations"

    # Model for vectorization
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Path to the PDF
    pdf_path = '/Users/saahithkalakuntla/Desktop/indian_resturant.pdf'

    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    # Extract and process text from PDF
    menu_text = extract_text_from_pdf(pdf_path)
    menu_items = parse_menu_text(menu_text)
    vectors = vectorize_items(menu_items, model)

    # Upload vectors to Pinecone
    upload_vectors_to_pinecone(vectors, index_name, api_key, environment)

if __name__ == "__main__":
    main()

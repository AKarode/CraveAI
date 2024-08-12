import fitz  # PyMuPDF
import re

# Step 1: Extract Text from the PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
    except Exception as e:
        print(f"An error occurred: {e}")
    return text

# Update the pdf_path to point to your PDF on the desktop
pdf_path = '/Users/saahithkalakuntla/Desktop/indian_resturant.pdf'  # Update this path if necessary

# Step 2: Parse the Extracted Text
def parse_menu_text(menu_text):
    menu_items = []
    lines = menu_text.split('\n')
    current_item = None
    description_lines = []

    for line in lines:
        # Remove leading and trailing whitespace
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Check if the line contains a price
        match = re.search(r'(\$[0-9.]+)', line)
        if match:
            price = match.group(1).strip()
            if current_item:
                current_item['price'] = price
                current_item['description'] = ' '.join(description_lines).strip()
                menu_items.append(current_item)
                current_item = None
                description_lines = []
        else:
            # If the line does not contain a price, it's part of the name or description
            if current_item is None:
                current_item = {'name': line, 'description': '', 'price': ''}
            else:
                description_lines.append(line)

    return menu_items

# Verify if the file exists
import os
if not os.path.exists(pdf_path):
    print(f"File not found: {pdf_path}")
else:
    menu_text = extract_text_from_pdf(pdf_path)
    print("Extracted Text from PDF:")
    print(menu_text)
    print("\n-----------------------\n")

    menu_items = parse_menu_text(menu_text)
    print("Parsed Menu Items:")
    for item in menu_items:
        print(f"Name: {item.get('name', '')}, Description: {item.get('description', '')}, Price: {item.get('price', '')}")
    print("\n-----------------------\n")

# Vectorization and Pinecone upload steps will follow after verification

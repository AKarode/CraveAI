import fitz  # PyMuPDF
import re
import os
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

# Main execution for checking text parsing
def main():
    # Path to the PDF
    pdf_path = '/Users/saahithkalakuntla/Desktop/indian_menu.pdf'

    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    # Extract and process text from PDF
    menu_text = extract_text_from_pdf(pdf_path)
    menu_items = parse_menu_text(menu_text)
    print(menu_items)

if __name__ == "__main__":
    main()

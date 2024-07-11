import os
import csv
from openai import OpenAI

# openai.api_key = ""
client = OpenAI()

response = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "I want you to read through this entire menu given line by line and understand all the food and drink items, and understand that each have an according price to it. I want you to then list the food and drink items in this menu by listing each menu item from lowest to highest price and dont include any message before it. Format it like this Item1 1.99\nItem2 2.49\nItem3 3.50\nItem4 4.00\n "},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://images.squarespace-cdn.com/content/v1/53c6cb4fe4b04eee7bdbd664/22afb33b-fb6f-4661-acf7-cb8d508e5b43/TAJ-DINE-IN-FRONT.jpg",
          },
        },
      ],
    }
  ],
  max_tokens=400,
)

# Print the full response to understand its structure

menu_text = response.choices[0].message.content

print(menu_text)

lines = menu_text.split('\n')

# Open a CSV file to write the data
with open('sorted_menu.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write the header
    csvwriter.writerow(['Item', 'Price'])
    
    # Process each line
    for line in lines:
        # Split each line by the dollar sign
        item, price = line.rsplit(' ', 1)
        # Write the item and price to the CSV file
        csvwriter.writerow([item.strip(), f'${price.strip()}'])

print("CSV file 'sorted_menu.csv' created successfully.")


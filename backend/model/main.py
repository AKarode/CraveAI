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
        {"type": "text", "text": "Can you sort the items in this image by listing each menu item from lowest to highest price? After you are done sorting, convert the text into a csv file"},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://mobile-cuisine.com/wp-content/uploads/2022/12/In-N-Out-Burger-menu-prices.jpg",
          },
        },
      ],
    }
  ],
  max_tokens=300,
)

# Print the full response to understand its structure
print(response)

# Extract the content from the response
response_content = response.choices[0].message.content

# Assuming the response is a text with items and prices separated by new lines
# and that the CSV formatted part starts after the text description.
csv_data_start = response_content.find("Item,Price")
csv_data = response_content[csv_data_start:]

# Split the CSV data into lines
lines = csv_data.split('\n')

# Skip the header line and create a list of dictionaries for CSV writing
data = []
for line in lines[1:]:  # Skip the header
    if line.strip():  # Skip any empty lines
        name, price = line.split(',')
        data.append({'Item': name.strip(), 'Price': price.strip()})

# Define the CSV file name
csv_file = 'sorted_menu.csv'

# Write the data to a CSV file
with open(csv_file, 'w', newline='') as csvfile:
    fieldnames = ['Item', 'Price']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in data:
        writer.writerow(row)

print(f'CSV file "{csv_file}" has been generated successfully.')

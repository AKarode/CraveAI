# gmg-food-assistant
(Tech Stacks Used)
- ReactJS
- PineconeDB
- OpenAI API
- Python

# Problem Statement
In the modern dining experience, restaurants often struggle to make their menus easily accessible and personalized for their customers. With menus frequently presented in PDF or image formats, customers need help navigating through various options and making informed decisions quickly, especially when faced with language barriers or unfamiliar dishes. Restaurants also lack a systematic way to gather and use customer feedback on specific dishes, missing out on valuable data to improve their offerings.

# Solution Using AI
GMG Food Assistant leverages AI to revolutionize how restaurants manage and present their menus. By using OCR (Optical Character Recognition) and Natural Language Processing (NLP), the system automatically extracts menu items from PDFs or images, making the data more accessible and usable. With the integration of PineconeDB for Retrieval-Augmented Generation (RAG), the AI intelligently matches menu items to user preferences, providing personalized and contextually relevant dish recommendations. Additionally, the system bridges language gaps by supporting Indic language translation, allowing restaurants to cater to a wider audience. Through AI-powered recommendations and a feedback loop, GMG Food Assistant helps restaurants refine their menu offerings and enhances the overall dining experience by making it more personalized, seamless, and data-driven.



## Version-1
- Restaurant uploads menu
- Our model _extracts_ menu items from pdf/png. (Use OCR to scan for characters, if language is Indic we can try use translators).
- Create relational database based off menu items and have it relate items to food that it is trained on (Pinecone DB to use for RAG context usage)
- Fine tune model to make sure exclusively give items from menus provided.

## Model Installation 
```bash
git clone
```
```bash
pip install openai
```

## Frontend Installation
```bash
git clone
```
```bash
npm install --legacy-peer-deps
```
```bash
npm run ios
```

## Contribution Rules
- Any edits made should be on new branch, main is left untouched until someone else tests it.

## Features to Add
- After items are recommended to user, user selects item they pick. This will be stored as a 1 in backend if the item is picked, 0 if the item wasn't picked, or NA if this data is unavaliable.
- After, maybe about 30 minutes, notification sent to user on whether they enjoyed the item they picked. This will also be stored as a 1 in backend if the user did enjoy the item, 0 if they said they didn't and NA if the data is unavaliable

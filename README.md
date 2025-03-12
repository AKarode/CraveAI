# CraveAI

<p align="center">
  <img src="frontend/public/logo.png" alt="CraveAI Logo" width="200"/>
</p>

CraveAI is an intelligent food assistant that revolutionizes the restaurant menu experience by using AI to extract menu items, provide personalized recommendations, and bridge language barriers.

## ✨ Features

- **Menu Extraction**: Automatically extracts menu items from PDF or image files using OCR
- **Personalized Recommendations**: Uses AI to provide contextually relevant dish recommendations
- **Multilingual Support**: Supports translation for Indic languages
- **User Feedback Collection**: Gathers and analyzes user feedback to improve recommendations
- **Restaurant Dashboard**: Helps restaurants manage menus and understand customer preferences

## 🛠️ Tech Stack

### Frontend
- **Framework**: Next.js 15.x (React 19)
- **Styling**: Tailwind CSS
- **UI Components**: Shadcn/UI
- **PDF Processing**: React-PDF
- **Image Processing**: Tesseract.js
- **State Management**: Zustand

### Backend
- **Framework**: FastAPI
- **AI/ML**: OpenAI API
- **Vector Database**: PineconeDB (for RAG)
- **OCR**: Tesseract/OCR Services
- **Containerization**: Docker

## 🚀 Getting Started

### Prerequisites

- Node.js (v18+ recommended)
- Python 3.10+
- Docker and Docker Compose (optional, for backend)
- OpenAI API key
- PineconeDB API key

### Backend Setup

**Option 1: Using Docker (recommended)**
```bash
# Navigate to the backend directory
cd backend

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys

# Build and run with Docker Compose
docker-compose up
```

**Option 2: Manual Setup**
```bash
# Navigate to the backend directory
cd backend

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys

# Start the FastAPI server
uvicorn app:app --reload
```

The backend API will be available at http://localhost:8000

### Frontend Setup

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at http://localhost:3000

## 📖 Usage

### Uploading a Menu
1. Log in to your restaurant account
2. Navigate to the "Menu Management" section
3. Click "Upload Menu" and select a PDF or image file
4. The system will automatically extract menu items
5. Review and edit extracted items if needed, then save

### User Experience
1. Browse restaurant listings
2. Select a restaurant to view its menu
3. Use the AI assistant to get personalized recommendations
4. Select dishes and provide feedback on your experience

## 🔄 How It Works

1. **Menu Extraction**: When a restaurant uploads a menu, our OCR service extracts text from the document. For non-English menus, translation services convert the text to English.

2. **Vector Database**: Extracted menu items are processed and stored in PineconeDB, creating embeddings that capture the semantic meaning of each dish.

3. **Recommendation Engine**: When users interact with the system, their preferences and queries are matched against the menu embeddings to provide personalized recommendations.

4. **Feedback Loop**: User selections and satisfaction ratings are collected to continuously improve the recommendation algorithm.

## 📚 API Documentation

The backend API documentation is available at http://localhost:8000/docs when the backend is running.

Key endpoints include:
- `/upload-menu`: Upload and process restaurant menus
- `/recommendations`: Get personalized dish recommendations
- `/feedback`: Submit user feedback on recommendations

## 🤝 Contributing

We welcome contributions to CraveAI! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

Please make sure to update tests as appropriate and adhere to the existing coding style.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Team

- Adit Karode - Lead Developer and Founder
- Anish - Frontend Developer 
- Mukhul - Frontend Developer 
- Rushil - Machine Learning Engineer 
- Saahith - Machine Learning Engineer 

## 🙏 Acknowledgements

- [OpenAI](https://openai.com) for their powerful API
- [PineconeDB](https://www.pinecone.io) for vector search capabilities
- [Tesseract.js](https://tesseract.projectnaptha.com) for OCR functionality

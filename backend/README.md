# Crave AI Backend

This is the backend service for Crave AI, a food recommendation system that provides personalized menu recommendations using OCR and AI.

## Technologies Used

- Python 3.10+
- FastAPI
- Pinecone for vector storage and retrieval
- OpenAI API for NLP
- Docker for containerization

## Getting Started

### Local Development

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in your API keys and configuration
5. Run the development server:
   ```bash
   python app.py
   ```

### Docker

1. Build the Docker image:
   ```bash
   docker build -t crave-ai-backend .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 --env-file .env crave-ai-backend
   ```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Deployment

For AWS deployment, the container can be deployed to:
- Amazon ECS
- Amazon EKS
- AWS App Runner
- AWS Elastic Beanstalk with Docker support 
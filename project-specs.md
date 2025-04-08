# CraveAI Project Specifications

## Project Overview

CraveAI is a personalized food recommendation system that leverages artificial intelligence, knowledge graphs, and retrieval-augmented generation (RAG) to provide contextually relevant dish recommendations to users. The system processes restaurant menus using OCR technology, understands user preferences and dietary restrictions, and delivers accurate, personalized food recommendations.

## Core Objectives

- Provide highly personalized food recommendations based on user preferences
- Process and extract structured data from restaurant menus (PDF/images)
- Support multilingual menus and user interactions
- Create a knowledge graph of food-related entities and relationships
- Enable natural language queries about food and restaurant offerings
- Collect and utilize user feedback to improve recommendations

## System Architecture

### High-Level Architecture

```
┌────────────────┐     ┌─────────────────┐     ┌────────────────────┐
│                │     │                 │     │                    │
│  Next.js       │────▶│  FastAPI        │────▶│  Vector Database   │
│  Frontend      │     │  Backend        │     │  (PineconeDB)      │
│                │     │                 │     │                    │
└────────────────┘     └─────────────────┘     └────────────────────┘
                               │                          │
                               ▼                          │
          ┌──────────────────────────────────┐           │
          │                                  │           │
          │  LLM Services (OpenAI)          │◀──────────┘
          │                                  │
          └──────────────────────────────────┘
                               │
                               ▼
          ┌──────────────────────────────────┐
          │                                  │
          │  Knowledge Graph                 │
          │  (Neo4j or In-memory Graph)      │
          │                                  │
          └──────────────────────────────────┘
```

### Backend Components

1. **API Layer (FastAPI)**
   - Handles HTTP requests and responses
   - Manages authentication and authorization
   - Coordinates processing flows

2. **OCR Service**
   - Processes PDF and image menus
   - Extracts text and structured menu items
   - Handles multilingual content

3. **Recommendation Service**
   - Stores and retrieves menu items from vector database
   - Processes user preferences and dietary restrictions
   - Generates personalized recommendations

4. **Graph RAG Service**
   - Maintains a knowledge graph of food entities and relationships
   - Enhances recommendations with graph-based reasoning
   - Answers food-related questions using graph and LLM

### Frontend Components

1. **Next.js Application**
   - React-based UI with Tailwind CSS and Shadcn/UI
   - Restaurant and menu browsing
   - User preference management
   - Recommendation display

2. **PDF/Image Processing**
   - Client-side PDF rendering with React-PDF
   - Optional image preprocessing with Tesseract.js

## Data Models

### Core Entities

1. **User**
   - Preferences (taste profile)
   - Dietary restrictions and allergies
   - Past interactions and feedback

2. **Restaurant**
   - Name, location, cuisine type
   - Operating hours
   - Rating and reviews

3. **Menu**
   - Restaurant association
   - List of menu items
   - Raw text and processed structure

4. **Menu Item**
   - Name and description
   - Price
   - Ingredients
   - Tags (vegetarian, spicy, etc.)
   - Vector embeddings

5. **Knowledge Graph**
   - Food entities (ingredients, dishes, cuisines)
   - Relationships (contains, similar-to, pairs-with)
   - Properties (taste profile, nutritional info)

## API Specifications

### Menu Processing

**Endpoint:** `POST /api/process-menu`

**Request:**
- File upload (PDF, PNG, JPG, JPEG)

**Response:**
```json
{
  "menu_id": "uuid-string",
  "items": [
    {
      "name": "Dish Name",
      "description": "Dish description",
      "price": "price-string",
      "tags": ["tag1", "tag2"]
    }
  ],
  "success": true,
  "message": "Success message"
}
```

### Recommendations

**Endpoint:** `POST /api/recommendations/{menu_id}`

**Request:**
```json
{
  "preferences": ["spicy", "vegetarian", "italian"],
  "dietary_restrictions": ["gluten-free", "no-nuts"]
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "name": "Dish Name",
      "description": "Dish description",
      "price": "price-string",
      "match_score": 0.95,
      "tags": ["tag1", "tag2"]
    }
  ]
}
```

### Graph-Based Recommendations

**Endpoint:** `POST /api/graph-recommendations`

**Request:**
```json
{
  "query": "I want something spicy but not too heavy",
  "constraints": {
    "dietary_restrictions": ["vegetarian"],
    "price_range": {
      "min": 10,
      "max": 30
    }
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Success message",
  "recommendations": [
    {
      "name": "Dish Name",
      "description": "Dish description",
      "price": "price-string",
      "reasoning": "Explanation of why this was recommended",
      "tags": ["tag1", "tag2"]
    }
  ]
}
```

### Food Question Answering

**Endpoint:** `POST /api/food-question`

**Request:**
```json
{
  "question": "What pairs well with truffles?"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Success message",
  "answer": "Detailed answer to the question",
  "source_items": [
    {
      "name": "Reference dish or ingredient",
      "description": "Description"
    }
  ]
}
```

## Technology Stack

### Backend

- **Language:** Python 3.10+
- **Framework:** FastAPI
- **OCR:** Tesseract, pdfplumber
- **AI/ML:** OpenAI API
- **Vector DB:** PineconeDB
- **Knowledge Graph:** Neo4j or in-memory graph
- **Containerization:** Docker

### Frontend

- **Framework:** Next.js 15.x (React 19)
- **Styling:** Tailwind CSS
- **UI Components:** Shadcn/UI
- **PDF Processing:** React-PDF
- **Image Processing:** Tesseract.js
- **State Management:** Zustand

## Development Workflows

### Setup and Environment

1. **Backend Environment**
   - Python virtual environment with dependencies from requirements.txt
   - Environment variables in .env file (see .env.example for template)
   - Docker container for consistent development

2. **Frontend Environment**
   - Node.js with npm
   - Next.js development server
   - Environment variables in .env.local

### Development Process

1. **Feature Development**
   - Create feature branch from `dev`
   - Write tests first (TDD approach)
   - Implement feature
   - Submit PR to `dev` branch

2. **Code Quality**
   - Linting with flake8 and eslint
   - Formatting with black and prettier
   - Type checking with mypy and TypeScript

3. **Testing**
   - Backend: pytest for unit and integration tests
   - Frontend: Jest and React Testing Library

## Testing Strategy

### Backend Testing

1. **Unit Tests**
   - Service-level functionality
   - API endpoint behavior
   - Utility functions

2. **Integration Tests**
   - End-to-end API flows
   - Database interactions
   - External service integration

3. **Performance Tests**
   - API response times
   - Recommendation generation speed
   - OCR processing performance

### Frontend Testing

1. **Component Tests**
   - UI component rendering
   - User interactions
   - State management

2. **E2E Tests**
   - User flows
   - Integration with backend
   - Cross-browser compatibility

## Data Processing Pipeline

### Menu Processing

1. **Document Upload**
   - Accept PDF or image files
   - Validate file type and content

2. **OCR Processing**
   - Extract text from documents
   - Handle multilingual content
   - Process structured tables

3. **Menu Extraction**
   - Parse text into structured menu items
   - Identify dish names, descriptions, prices
   - Tag items with categories and attributes

4. **Vectorization**
   - Generate embeddings for menu items
   - Store in vector database for retrieval
   - Index for efficient searching

### Knowledge Graph Construction

1. **Entity Extraction**
   - Identify food entities (ingredients, dishes, cuisines)
   - Extract relationships between entities
   - Detect attributes and properties

2. **Graph Population**
   - Create nodes for entities
   - Establish relationships between nodes
   - Assign properties to nodes and relationships

3. **Graph Enrichment**
   - Integrate with external knowledge sources
   - Add nutritional information
   - Include taste profiles and pairing suggestions

## Deployment Strategy

### Development Environment

- Local development with Docker Compose
- Shared development environment for testing

### Staging Environment

- Cloud-based deployment (AWS, GCP, or similar)
- CI/CD pipeline with GitHub Actions
- Automated testing and deployment

### Production Environment

- Containerized deployment with Kubernetes or Docker Swarm
- Load balancing and auto-scaling
- Monitoring and alerting

## Monitoring and Analytics

### System Monitoring

- Server health and performance
- API endpoint performance
- Error tracking and logging (Sentry)

### Business Metrics

- User engagement
- Recommendation acceptance rate
- Search and query patterns

### Feedback Collection

- Explicit user ratings for recommendations
- Implicit feedback (clicks, time spent)
- A/B testing for recommendation algorithms

## Future Roadmap

### Phase 1: MVP Enhancement (Current)

- Improve OCR accuracy for diverse menu formats
- Enhance recommendation relevance
- Expand knowledge graph coverage

### Phase 2: Advanced Features

- Implement Model Context Protocol (MCP) for structured LLM prompts
- Add multilingual translation capabilities
- Develop user preference learning algorithms

### Phase 3: Scale and Optimization

- Optimize vector search for large menu databases
- Implement caching strategies for common queries
- Enhance graph traversal efficiency

### Phase 4: Extended Capabilities

- Add nutritional analysis and health-focused recommendations
- Develop restaurant-specific insights dashboard
- Implement seasonal and trending recommendations

## Implementation Guidelines

### Code Organization

- Modular architecture with clear separation of concerns
- Service-oriented design for backend components
- Component-based architecture for frontend

### Documentation

- API documentation with OpenAPI/Swagger
- Code documentation with docstrings
- Architecture and design decisions in project wiki

### Security

- API authentication and authorization
- Data encryption for sensitive information
- Regular security audits and vulnerability scanning

## Conclusion

CraveAI aims to transform the restaurant dining experience by providing personalized, context-aware food recommendations. By combining OCR technology, AI-driven recommendations, and knowledge graph reasoning, the system offers unique value to both users and restaurants.

This project specification provides a foundation for development, but should be treated as a living document that evolves as the project progresses and requirements change. 
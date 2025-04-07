# CraveAI Backend Tests

This directory contains tests for the CraveAI backend services organized according to industry standards.

## Test Structure

```
tests/
├── unit/                   # Unit tests for all modules
│   ├── services/           # Tests for services directory modules
│   │   ├── test_ai_recommendations.py
│   │   ├── test_ocr.py
│   └── utils/              # Tests for utility functions (future)
├── integration/            # Integration tests
│   ├── test_openai.py      # OpenAI API integration tests  
│   └── test_pinecone.py    # Pinecone integration tests
├── api/                    # API endpoint tests
│   └── test_api_endpoints.py
├── e2e/                    # End-to-end tests (future)
├── performance/            # Performance/load tests
│   └── model_comparison/   # Model comparison tests
├── fixtures/               # Test fixtures and mocks
│   └── menu_samples/       # Sample menus for testing
├── conftest.py             # Pytest configuration and shared fixtures
└── README.md               # This file
```

## Running Tests

### Unit Tests

```bash
cd backend
pytest tests/unit -v
```

### Integration Tests

```bash
cd backend
pytest tests/integration -v
```

### API Tests

```bash
cd backend
pytest tests/api -v
```

### All Tests

```bash
cd backend
pytest
```

### With Coverage Report

```bash
cd backend
pytest --cov=services --cov-report=term-missing
```

## Test Fixtures

Common test fixtures are defined in `conftest.py`, including:

- `api_keys`: Provides API keys for external services
- `openai_client`: Provides an OpenAI client
- `pinecone_client`: Provides a Pinecone client
- `mock_openai_embedding`: Mocks OpenAI embedding API
- `sample_menu_items`: Sample menu data for testing

## Model Comparison Tests

The model comparison tests are in the performance directory:

```bash
cd backend
python -m tests.performance.model_comparison.run_model_comparison
```

For real menu tests (using the Cheesecake Factory menu):

```bash
cd backend
python -m tests.performance.model_comparison.run_real_menu_comparison
```

## Writing New Tests

When adding new tests, follow these guidelines:

1. Place tests in the appropriate directory based on test type
2. Use pytest fixtures from conftest.py when possible
3. Follow the naming convention: `test_*.py` for files, `test_*` for functions
4. Mock external dependencies to avoid real API calls in unit tests
5. Keep tests independent and idempotent

## Code Coverage

Aim for at least 80% code coverage for all services and critical paths.

Run coverage report:

```bash
cd backend
pytest --cov=services --cov-report=html
```

Then view the HTML report in `htmlcov/index.html`.
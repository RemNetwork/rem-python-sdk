# Contributing to rem-vectordb

Thank you for considering contributing to the REM Python SDK!

## Development Setup

```bash
# Clone the repo
git clone https://github.com/RemNetwork/rem-python-sdk.git
cd rem-python-sdk

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install in development mode with all extras
pip install -e ".[langchain,llamaindex]"
pip install pytest pytest-asyncio ruff
```

## Running Tests

```bash
pytest tests/ -v
```

Tests use `httpx.MockTransport` to avoid hitting the real API.

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check rem/ tests/
ruff format rem/ tests/
```

## Making Changes

1. Fork the repo and create a feature branch
2. Make your changes
3. Add tests for new functionality
4. Run `pytest` and `ruff check` to verify
5. Submit a pull request

## Release Process

Releases are automated via GitHub Actions:

1. Update version in `pyproject.toml` and `rem/__init__.py`
2. Commit and tag: `git tag v0.x.0`
3. Push: `git push origin main --tags`
4. CI runs tests, then publishes to PyPI

## Project Structure

```
rem/
├── __init__.py          # Package exports
├── client.py            # REM and AsyncREM clients
├── collection.py        # Collection and AsyncCollection
├── types.py             # Pydantic models
├── exceptions.py        # Error types
└── integrations/
    ├── langchain.py     # LangChain vector store
    └── llamaindex.py    # LlamaIndex vector store
tests/
└── test_sdk.py          # Test suite
examples/
├���─ quickstart.py        # Basic usage
└── async_example.py     # Async patterns
```

## Questions?

- Discord: https://discord.gg/9ndMQY4PYP
- Email: support@getrem.online

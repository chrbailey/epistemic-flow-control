# Contributing to Epistemic Flow Control

Thank you for your interest in contributing! This project aims to make LLM outputs reliable for high-stakes decisions through human oversight.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/chrbailey/epistemic-flow-control.git
   cd epistemic-flow-control
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[all]"
   ```

4. **Run the tests**
   ```bash
   pytest tests/ -v
   ```

## Making Changes

### Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check . --fix
```

### Type Hints

Type hints are encouraged but not strictly required. Run mypy to check:

```bash
mypy core gates validation training --ignore-missing-imports
```

### Testing

- All new features should include tests
- Tests go in the `tests/` directory
- Use pytest fixtures for shared setup
- Mark slow tests with `@pytest.mark.slow`
- Mark LLM-dependent tests with `@pytest.mark.llm`

Run tests:
```bash
# All tests
pytest tests/ -v

# Skip slow tests
pytest tests/ -v -m "not slow"

# With coverage
pytest tests/ -v --cov=core --cov-report=html
```

## Pull Request Process

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** with clear, descriptive commits
3. **Add tests** for any new functionality
4. **Update documentation** if needed
5. **Run the full test suite** before submitting
6. **Open a pull request** with a clear description

### PR Title Convention

Use clear, descriptive titles:
- `feat: Add temporal decay visualization`
- `fix: Correct Wilson score calculation for edge case`
- `docs: Improve quick start guide`
- `test: Add integration tests for review gate`

## Architecture Overview

Understanding the system helps you contribute effectively:

```
Events (Ground Truth)
    ↓
Pattern Extraction (LLM + Human Validation)
    ↓
Pattern Database (Bayesian Weights)
    ↓
Predictions (Calibrated Confidence)
    ↓
Review Gate (Human Approval)
    ↓
Production Output
    ↓
Outcomes → Training Data → Improved Calibration
```

### Key Design Principles

1. **Humans control the flow** - Every high-stakes decision has a human gate
2. **Ground truth from events** - All patterns trace back to verifiable events
3. **Bayesian updating** - Confidence grows with evidence
4. **Calibrated confidence** - Predictions match actual accuracy
5. **Temporal decay** - Old patterns fade without fresh evidence

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas

Thank you for contributing!

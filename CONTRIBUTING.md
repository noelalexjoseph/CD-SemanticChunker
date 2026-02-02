# Contributing to Construction RAG

Thank you for your interest in contributing to Construction RAG! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/construction-rag.git
   cd construction-rag
   ```
3. Install development dependencies:
   ```bash
   pip install -e .[dev]
   ```

## Development Setup

### Running Tests

```bash
pytest tests/
```

### Code Formatting

We use Black for code formatting and Ruff for linting:

```bash
black src/
ruff check src/
```

### Type Checking

```bash
mypy src/construction_rag/
```

## Making Changes

1. Create a new branch for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure tests pass

3. Update documentation if needed

4. Commit your changes with a clear message:
   ```bash
   git commit -m "Add feature: description of your changes"
   ```

5. Push to your fork and create a Pull Request

## Pull Request Guidelines

- Keep PRs focused on a single feature or fix
- Include tests for new functionality
- Update documentation as needed
- Follow the existing code style
- Write clear commit messages

## Code Style

- Use type hints for function signatures
- Write docstrings for public functions and classes
- Follow PEP 8 guidelines
- Keep lines under 100 characters

## Reporting Issues

When reporting issues, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages (if any)

## Questions?

Feel free to open an issue for questions or discussions about the project.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

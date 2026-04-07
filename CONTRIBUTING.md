# Contributing Guide

Thank you for considering a contribution.

## Development setup

1. Create and activate a virtual environment.
2. Install editable package with development tools:

```bash
pip install -e .[dev]
```

3. Run local checks before opening a pull request:

```bash
pytest
ruff check .
mypy src
```

## Contribution workflow

1. Open an issue describing the bug/feature.
2. Create a branch from main.
3. Add tests for behavior changes.
4. Keep docs updated (`README`, `LEARN`, or docs files).
5. Submit a pull request with clear summary and rationale.

## Pull request expectations

- New features include tests.
- Breaking changes are explicitly documented.
- Public interfaces include docstrings.
- Configuration and artifact changes include migration notes.

## Coding standards

- Python 3.10+
- Type hints on public APIs
- Small functions and explicit names
- Avoid hidden side effects

# Contributing

Thanks for your interest in contributing! This guide covers everything you need to get started.

## Development Setup

```bash
git clone https://github.com/<org>/cinematic-controlnet.git
cd cinematic-controlnet
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
ruff check .
ruff format .
```

Please ensure all code passes `ruff check` and `ruff format --check` before submitting.

## Submitting a Pull Request

1. Fork the repository and clone your fork.
2. Create a feature branch from `main`:
   ```bash
   git checkout -b my-feature
   ```
3. Make your changes, add tests where appropriate.
4. Ensure tests pass and code is formatted.
5. Push to your fork and open a pull request against `main`.

### PR Guidelines

- Keep PRs focused on a single change.
- Write a clear description of what the PR does and why.
- Link any related issues.
- Ensure CI passes before requesting review.

## Reporting Issues

Use the GitHub issue templates for bug reports and feature requests.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

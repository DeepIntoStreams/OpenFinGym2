# Developers Guide

## Code Formatting

Code formatting checks can be run with

```commandline
uv run pre-commit run --all-files
```

## Tests

Tests can be run with

```commandline
uv run pytest
```

## Documentation

Documentation is built using [Sphinx](https://www.sphinx-doc.org/en/master/).
Documentation build files are located in `/docs`, and can be built with

```commandline
uv run task docs
```

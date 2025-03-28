# Contributing to DSPy Meme Generator

First off, thank you for considering contributing to DSPy Meme Generator! It's people like you that make this project such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [conduct@example.com](mailto:conduct@example.com).

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps which reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include screenshots and animated GIFs if possible
* Include error messages and stack traces
* Include the version of the project you're using
* Include your environment details (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* Use a clear and descriptive title
* Provide a step-by-step description of the suggested enhancement
* Provide specific examples to demonstrate the steps
* Describe the current behavior and explain which behavior you expected to see instead
* Explain why this enhancement would be useful
* List some other applications where this enhancement exists, if applicable

### Pull Requests

Please follow these steps to have your contribution considered by the maintainers:

1. Follow all instructions in the template
2. Follow the style guides
3. After you submit your pull request, verify that all status checks are passing

#### Pull Request Process

1. Update the README.md with details of changes to the interface, if applicable
2. Update the documentation with any new features or changes
3. The PR will be merged once you have the sign-off of at least one maintainer

## Development Process

### Setting Up Development Environment

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/dspy-meme-gen.git
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/original/dspy-meme-gen.git
   ```
4. Install development dependencies:
   ```bash
   poetry install --with dev
   ```
5. Install pre-commit hooks:
   ```bash
   poetry run pre-commit install
   ```

### Development Workflow

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Run tests:
   ```bash
   poetry run pytest
   ```
4. Run code quality checks:
   ```bash
   poetry run pre-commit run --all-files
   ```
5. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: add your feature"
   ```
6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
7. Create a Pull Request

### Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

* `feat:` - A new feature
* `fix:` - A bug fix
* `docs:` - Documentation only changes
* `style:` - Changes that do not affect the meaning of the code
* `refactor:` - A code change that neither fixes a bug nor adds a feature
* `perf:` - A code change that improves performance
* `test:` - Adding missing tests or correcting existing tests
* `chore:` - Changes to the build process or auxiliary tools

Example:
```
feat(meme-generation): add support for custom templates

- Add template validation
- Implement template storage
- Update documentation
```

### Code Style Guide

#### Python Style Guide

* Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
* Use type hints for all function arguments and return values
* Write descriptive docstrings following [PEP 257](https://www.python.org/dev/peps/pep-0257/)
* Maximum line length is 88 characters (Black default)
* Use f-strings for string formatting
* Use `pathlib` for file path operations

#### Testing Guidelines

* Write tests for all new features
* Maintain test coverage above 90%
* Use descriptive test names that explain the behavior being tested
* Use fixtures for common test setup
* Mock external services appropriately

Example test:
```python
def test_meme_generation_with_custom_template(mock_openai: MockerFixture) -> None:
    """
    Test meme generation with a custom template.
    
    Args:
        mock_openai: Pytest fixture for mocking OpenAI API calls
    """
    # Test implementation
```

### Documentation Guidelines

* Keep README.md up to date
* Document all public APIs
* Include examples in docstrings
* Update CHANGELOG.md for all notable changes
* Write clear commit messages
* Add comments for complex logic

## Additional Notes

### Issue and Pull Request Labels

* `bug` - Something isn't working
* `enhancement` - New feature or request
* `documentation` - Improvements or additions to documentation
* `good first issue` - Good for newcomers
* `help wanted` - Extra attention is needed
* `invalid` - Something's wrong
* `question` - Further information is requested
* `wontfix` - This will not be worked on

## Recognition

Contributors will be recognized in:
* The project's README.md
* Our documentation website
* Release notes when their contributions are included

## Questions?

If you have questions, please feel free to:
* Open an issue
* Join our Discord server
* Email the maintainers

Thank you for contributing to DSPy Meme Generator! 
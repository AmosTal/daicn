# Contributing to Decentralized AI Computation Network (DAICN)

## Welcome Contributors! 🌟

We're excited that you're interested in contributing to DAICN. This document provides guidelines for contributing to the project.

### 🤝 How to Contribute

1. **Fork the Repository**
   - Create a personal fork of the project on GitHub
   - Clone your fork to your local machine

2. **Set Up Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

### 🧪 Development Guidelines

#### Code Standards
- Follow PEP 8 Python style guidelines
- Use type hints
- Write comprehensive docstrings
- Maintain 90%+ test coverage

#### Testing
- Run tests before submitting a PR
  ```bash
  pytest tests/
  ```
- Add new tests for any new functionality

#### Commit Message Convention
- Use descriptive, concise commit messages
- Format: `<type>(<scope>): <description>`
  - Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
  - Example: `feat(dashboard): add network health visualization`

### 🚀 Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Add description of changes in PR
4. Wait for code review

### 🛡️ Code of Conduct

- Be respectful and inclusive
- Collaborate constructively
- Prioritize project goals
- No harassment or discrimination

### 🔍 Areas of Focus

- Blockchain integration
- Performance optimization
- Security enhancements
- Machine learning task allocation
- Frontend improvements

### 📞 Contact

For questions, please open an issue or contact [project maintainer email].

### 📄 License

By contributing, you agree to license your contributions under the MIT License.

**Thank you for helping democratize AI computation!** 🌐🤖

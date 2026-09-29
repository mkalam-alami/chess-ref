# Agent Development Guidelines

## Default Branch Policy

**All agents must work on `main` unless explicitly specified otherwise.**

This ensures that:
- Changes are immediately available to the team
- Reduces branch proliferation and merge conflicts
- Simplifies integration and deployment workflows
- Maintains a single source of truth

### Exceptions

Development branches may be used only when explicitly requested in:
- Project specifications
- Task descriptions
- Explicit user instructions

When an exception applies, the task description will clearly state the target branch name.

## Commit Guidelines

- Write clear, descriptive commit messages explaining the "why"
- Include any relevant context or rationale
- Keep commits focused and atomic when possible

## Code Review

All changes pushed to main should follow the repository's existing code standards and patterns.

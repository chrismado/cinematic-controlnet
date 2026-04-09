# Security Policy

## Reporting a Vulnerability

Please use a private GitHub security advisory or contact the maintainer directly when reporting a vulnerability. Include the affected module, reproduction steps, and any environment-specific details needed to validate the report.

## Audit Summary

- No hardcoded credentials, access tokens, or API secrets were found during the April 2026 security audit.
- No dangerous `eval()` usage, unsafe YAML loaders, or user-controlled shell execution paths were found in the codebase.
- The training and inference flows operate on local tensors and checkpoints only; they do not accept SQL, shell, or pickle payloads from untrusted users.
- Re-run `pip-audit -r requirements.txt` and `bandit -r . -ll` whenever dependency or deployment changes are made.

# Team Workflow

## Branching

- Use branch naming: `feat/<owner>-<topic>`
- Open PRs early and review in pairs.

## Responsibility Split

- Member 1: preprocessing pipeline and data quality checks.
- Member 2: feature extraction pipeline and feature validation.
- Member 3: model architecture, training loops, and checkpoint strategy.
- Member 4: evaluation metrics, error analysis, and reporting.

## Definition of Done per Stage

- Code in `src/`
- Basic test in `tests/`
- Config in `configs/` (if relevant)
- Results snapshot in `reports/`

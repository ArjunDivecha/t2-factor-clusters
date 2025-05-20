# T2 Factor Clusters - Guidelines for Agentic Coding

## Environment Setup
- Python ≥3.10
- Required packages: pandas, numpy, scipy, scikit-learn, matplotlib, networkx, ruptures, pyarrow, fastparquet, zarr, python-louvain, loguru, scikit-bio

## Project Commands
- Setup: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
- Run analysis: `python <script_name>.py --config config.yml`
- Lint: `flake8 . --max-line-length=100 --exclude=venv`
- Type check: `mypy --ignore-missing-imports .`

## Coding Style
- PEP 8 compliant with 100 character line limit
- Use type hints throughout (Python's typing module)
- Snake_case for variables/functions, PascalCase for classes
- Import order: standard library → third party → local modules
- Error handling: use try/except with specific exceptions
- Logging: use loguru for all logging (not print statements)

## Data Organization
- Raw data in data/raw/
- Processed data in data/processed/
- Derived outputs in data/derived/
- Visualization outputs in reports/
- AI-uploadable outputs in outputs/upload_bundle/

## Version Control
- Meaningful commit messages describing purpose
- Branch for new features, merge when complete
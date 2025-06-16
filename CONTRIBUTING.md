

## Installation

1. Clone the repo:
   ```bash
   git clone git@github.com:your-username/validityBase-project.git
   cd validityBase-project
Install package and dev dependencies:
pip install -e .[dev]
# or
pip install -r requirements_dev.txt
Usage

Build the SPY500 factor panel:

# From a local returns CSV:
python -m vbase_utils.factors.spy500 \
  --csv path/to/us_stocks_1d_rets.csv \
  --out spy500_panel.parquet

# Or download fresh data via yfinance:
python -m vbase_utils.factors.spy500 \
  --yf 2018-01-01 2025-06-10 \
  --out spy500_panel.parquet
Testing

Run the unit tests:

pytest
Style & Workflow

Code lives under src/vbase_utils/...
Tests live under tests/..., filenames must start with test_
Commit messages are short imperatives, e.g. “Add spy500 factor module”
Work in feature branches; open a Pull Request against main (or dev) when ready
CI will run pre-commit hooks, black, isort, and pytest automatically

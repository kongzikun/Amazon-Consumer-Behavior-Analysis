# Data Directory

This folder stores the raw assets required to reproduce the analysis.

## Files
- `amazon-purchases.csv` – Raw Amazon transaction history (~300 MB). Download from the Open E-commerce 1.0 data release and place it in this directory. The file is ignored by Git because of its size.
- `survey.csv` – Household survey responses included as a lightweight sample in the repository.

## Usage
The scripts in `src/` default to reading from `data/amazon-purchases.csv` and `data/survey.csv`. You can override these paths when instantiating the data processors if you wish to keep the raw data elsewhere.

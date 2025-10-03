# Amazon Consumer Behavior Analysis

Amazon Consumer Behavior Analysis is an end-to-end data exploration project built on the **Open E-commerce 1.0** dataset. It combines rigorous preprocessing with a library of interactive dashboards and publication-ready figures that describe how households interact with Amazon across demographics, product categories, and time.

## Highlights
- Clean and merge large-scale purchase and survey data with configurable preprocessing pipelines.
- Generate interactive Plotly dashboards covering demographics, temporal trends, behavioral segments, migration patterns, and COVID-19 impacts.
- Export static, publication-grade PNGs alongside the HTML dashboards for stakeholder reporting.
- Modular Python architecture that makes it easy to extend analyses or integrate new datasets.

## Project Structure
```
Amazon-Consumer-Behavior-Analysis/
├── assets/
│   └── images/                # High-resolution PNG exports
├── data/
│   └── survey.csv             # Survey responses (example data)
├── docs/
│   ├── analysis_notes.md      # Background notes captured during research
│   ├── prescreen-survey-instrument.pdf
│   └── survey-instrument.pdf
├── outputs/
│   ├── interactive/           # Interactive Plotly dashboards (HTML)
│   └── visualizations/        # Curated HTML storyboards and reports
├── src/
│   ├── amazon_consumer_analysis.py   # Core preprocessing + visualization classes
│   └── generate_static_visuals.py    # Batch export pipeline for PNG/HTML outputs
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Getting Started

### 1. Environment setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Data setup
1. Download **`amazon-purchases.csv`** from the Open E-commerce 1.0 dataset (the file is ~300 MB, so it is not tracked in this repository).
2. Place the file at `data/amazon-purchases.csv` alongside the provided `survey.csv`.
3. If you prefer to work with alternative locations, pass the file paths to the processors when instantiating the analysis classes.

### 3. Run the interactive pipeline
```bash
python src/amazon_consumer_analysis.py
```
This script merges the purchases and survey data, applies cleaning rules, and exposes helper classes for creating interactive dashboards. Use the class interfaces to build custom notebooks or scripts.

### 4. Regenerate publication-ready assets
```bash
python src/generate_static_visuals.py
```
This batch job recreates the curated PNG and HTML assets in `assets/images/` and `outputs/interactive/` respectively. Ensure that `kaleido` is installed (automatically handled via `requirements.txt`).

## Outputs
- Interactive Plotly dashboards for demographic, behavioral, and temporal analysis in `outputs/interactive/`.
- Static PNGs suitable for presentations and reports in `assets/images/`.
- Narrative HTML storyboards hosted under `outputs/visualizations/`.

## Contributing
Pull requests are welcome for:
- Additional segmentation logic or predictive models.
- Improved visual design or dashboard layout templates.
- Automated data refresh scripts and CI integrations.

## License
Released under the MIT License. See [LICENSE](LICENSE) for details.

## Citation
If you build upon this project, please cite the Open E-commerce 1.0 dataset and link back to this repository so others can reproduce your work.

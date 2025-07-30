# Amazon‑Consumer‑Behavior‑Analysis

**A multi‑faceted exploration of the Open E‑commerce 1.0 dataset (2018‑2022), revealing patterns in American digital consumer behaviour through temporal analysis, demographic segmentation and machine‑learning‑based customer archetypes.**

## Abstract

The project analyses the **Open E‑commerce 1.0** dataset, which consists of **1.8 million Amazon purchase records from over 5 000 U.S. shoppers between 2018 and 2022**, linked to their demographic profiles.  By cleansing and merging transactional data with survey responses, we build a rich view of consumer behaviour.  We find that spending patterns align with overall market performance and that pandemic restrictions accelerated online shopping while increasing heterogeneity.  Segmenting customers using an **RFM model and K‑Means clustering** uncovers distinct archetypes, such as **High‑Value Champions** and **At‑Risk** cohorts.  The dataset also offers potential for migration and socioeconomic studies.  Visual analytics drive each insight, emphasising the value of interactive dashboards for business and academic research.

## Key Features

* **Comprehensive demographic analysis** – charts illustrate gender, age, income and household size distributions.
* **Temporal breakdowns** – quarterly, monthly and daily patterns highlight spending and activity cycles; seasonal decomposition isolates trend and seasonality.
* **Behavioural insights** – heatmaps and scatter plots compare frequency, spending and product diversity for each user group.
* **Event‑driven analysis** – the impact of COVID‑19 and other key events is assessed through time‑stamped data and annotations.
* **Advanced segmentation** – an RFM model combined with K‑Means clustering identifies customer segments such as High‑Value Champions, Frequent Buyers and At‑Risk customers.
* **Geographical migration** – sankey diagrams and tenure analyses examine regional moves and tenure vs. state changes.
* **Publication‑ready visuals** – all charts are generated using Plotly and saved as high‑resolution PNG and interactive HTML files via a custom `save_academic_figure` function.

## Methodology

1. **Data loading & preprocessing:** Read the purchases and survey CSV files, convert dates and numeric fields, compute total amounts and filter out gift card transactions and outliers.
2. **Demographic profiling:** Merge purchase and survey data and visualise distributions of gender, age, income and household size.
3. **Temporal analysis:** Create quarterly spending trends, monthly comparisons of spending vs. purchase counts, daily heatmaps of activity and seasonal decomposition to separate trend and seasonality components.
4. **Behaviour analysis:** Explore relationships between purchase frequency, total spending and product category diversity; identify peak activity periods.
5. **Event analysis:** Assess the effects of the COVID‑19 pandemic and other key dates by annotating time series plots and examining related product categories.
6. **Advanced segmentation:** Use an **RFM (Recency, Frequency, Monetary)** model and **K‑Means clustering** to segment users; visualise segments with 3D scatter and normalized bar charts.
7. **Migration analysis:** Track customer movements across U.S. states and regions; employ sankey diagrams and scatter plots to understand tenure and spending patterns.

## Required Libraries

The analysis script relies on the following Python packages:

| Library | Purpose |
|--------|--------|
| `pandas`, `numpy` | Data loading, manipulation and numerical calculations |
| `plotly.express`, `plotly.graph_objects` | Interactive plotting and custom figure styling |
| `statsmodels` | Seasonal decomposition of time series |
| `scikit‑learn` | K‑Means clustering and data scaling (StandardScaler, MinMaxScaler) |
| `dash` (optional) | Can be used for building interactive dashboards |
| `warnings`, `os` | Utility modules for file management and warning suppression |

Ensure these libraries are installed (e.g., via `pip install pandas numpy plotly statsmodels scikit-learn`).

## Usage

1. **Prepare data:** Download the **Open E‑commerce 1.0** dataset and extract the purchases and survey CSV files.  Update the file paths when instantiating the `AmazonDataProcessor` class in `data_visualization_script.py`.
2. **Run the script:** Execute the script to load and process the data, generate visualisations and save them to the `images` and `interactive_html` directories.  For example:

   ```bash
   python data_visualization_script.py --purchases_path path/to/Purchases.csv --survey_path path/to/Survey.csv
   ```

3. **Explore outputs:** High‑resolution PNG images will be available under `images/` and interactive versions under `interactive_html/`.

## Team Members

This project was completed by **Group 2** (Project code SUSTech DS261p) as follows:

| Role | Contribution |
|-----|-------------|
| **Tianyi Lyu** (Project Lead) | Coordinated the project, defined research questions and oversaw all phases of analysis. |
| **Zikun Kong** (Data & Technical Lead) | Led data preprocessing, implemented visualisation scripts and integrated machine learning models. |
| **Zinan Ye** (Presentation Lead) | Designed the narrative flow, created the presentation and ensured clarity of figures and insights. |
| **Jinrui Zhao** (Presentation Co‑developer) | Assisted with presentation development and refining visual storytelling. |

## Citation

When using this repository or referencing its findings, please cite the original Open E‑commerce 1.0 dataset creators and this project's authors.  The analysis demonstrates how visual analytics can transform transactional datasets into strategic insights for both business and academic audiences.

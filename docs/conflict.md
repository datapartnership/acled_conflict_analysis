# ACLED Conflict Analysis Package

A Python package for extracting, processing, analyzing, and visualizing conflict data from the Armed Conflict Location & Event Data Project (ACLED).

## Package Overview

The `acled_conflict_analysis` package provides a comprehensive toolkit for working with ACLED conflict data. It follows Python best practices with a src-layout structure and modular design.

### Package Structure

```
acled_conflict_analysis/
├── LICENSE                      # Mozilla Public License 2.0
├── README.md                    # Package overview and quick start
├── CODE_OF_CONDUCT.md          # Community guidelines
├── CONTRIBUTING.md             # Contribution guidelines
├── pyproject.toml              # Package metadata and dependencies
├── setup.py                    # Build configuration
├── docs/                       # Documentation website
│   ├── _config.yml
│   ├── _toc.yml
│   └── notebooks/              # Tutorial notebooks
├── src/                        # Package source code
│   └── acled_conflict_analysis/
│       ├── __init__.py         # Package initialization
│       ├── acled_auth.py       # API authentication
│       ├── extraction.py       # Data extraction from ACLED API
│       ├── processing.py       # Data processing and spatial operations
│       └── visuals.py          # Static visualizations (matplotlib)
├── notebooks/                  # Example notebooks
│   ├── example.ipynb
│   ├── extraction.ipynb
│   └── visualizing-acled.ipynb
└── data/                       # Data storage (not in version control)
```

## Core Modules

### 1. `acled_auth.py` - Authentication

Handles secure API authentication for ACLED data access.

**Key Functions:**
- Credential management using environment variables
- Secure API key handling
- Authentication token generation

### 2. `extraction.py` - Data Extraction

Interfaces with the ACLED API to extract conflict event data.

**Key Functions:**
- `get_acled_data()`: Fetch conflict events by country, date range
- Handles API pagination and rate limiting
- Returns structured pandas DataFrames

### 3. `processing.py` - Data Processing

Transforms and aggregates conflict data for analysis.

**Key Functions:**
- `data_type_conversion()`: Clean and convert data types
- `convert_to_gdf()`: Convert to GeoPandas GeoDataFrame
- `get_acled_by_group()`: Temporal aggregation (daily, weekly, monthly)
- `convert_to_h3_grid()`: Spatial aggregation using H3 hexagons
- `get_acled_by_admin()`: Aggregate by administrative boundaries

**Spatial Operations:**
- H3 hexagonal binning for spatial aggregation (resolution 5 ≈ 276 km²)
- Geographic boundary processing
- Point-to-polygon spatial joins

### 4. `visuals.py` - Static Visualizations

Creates publication-quality matplotlib visualizations.

**Visualization Types:**
- `create_comparative_maps()`: Multi-panel comparison maps
- `get_h3_maps()`: Hexagonal choropleth maps with tercile binning
- `create_bivariate_map_with_basemap()`: Bivariate choropleth (events × fatalities)
- Bar charts and time-series plots
- Stacked bar charts for category comparison

**Features:**
- Open Sans typography for professional appearance
- Custom Mapbox basemap integration
- Tercile-based classification (equal observation count per bin)
- Customizable color schemes
- 16:9 aspect ratio (World Bank standard)

## Data Source: ACLED

### About ACLED

The [Armed Conflict Location & Event Data Project (ACLED)](https://acleddata.com/) is a disaggregated data collection, analysis, and crisis mapping project. ACLED collects information on the dates, actors, locations, fatalities, and types of all reported political violence and protest events around the world.

**Data Characteristics:**
- Coverage: Global, 75+ languages
- Frequency: Weekly updates
- Granularity: Point locations with coordinates
- Sources: Traditional media, reports, local partners, verified social media
- Methodology: Trained researchers, multi-source verification
- Quality Control: Retroactive corrections as new information emerges

### Data Collection Methodology

ACLED data are derived from a wide range of local, national, and international sources. The information is collected by trained researchers worldwide and coded weekly. During weekly coding, ACLED researchers find that many events are reported several times by multiple sources. The details of each report may differ, but ACLED researchers only extract factual information about the event: who was involved, where did it happen, when did it occur, what occurred, and in what sequence.

**Sourcing Approach:**

The sources used to identify conflict events include traditional media, reports, local partner data, and verified new media such as Twitter, WhatsApp, and Telegram. ACLED does not scrape data from social media but uses a targeted approach to the inclusion of new media through either the establishment of relationships with the source directly or the verification of the quality of each source.

**Source Attribution:**

Every ACLED event is based on at least one source. The source names or acronyms are noted in the 'Source' column. With the exception of certain local sources that wish to remain anonymous, the 'Source' column details are sufficient to retrace the sources that have been used to record an event. All sources listed have contributed information to the event. Researchers often find multiple reports confirming details about an event; when multiple sources report on the same information, the most thorough, reliable, and recent report is cited. The ACLED team corrects some of their past entries as they get new information about the reported conflict.

Additional details on ACLED's sourcing methodology: [ACLED Sourcing Methodology PDF](https://acleddata.com/acleddatanew/wp-content/uploads/dlm_uploads/2020/02/FAQs_ACLED-Sourcing-Methodology.pdf)

### Data Access

**Dataset Information:**
- **Dataset**: Armed Conflict Location & Event Data
- **Provider**: Armed Conflict Location & Event Data Project
- **Update Frequency**: Daily
- **Spatial Resolution**: Point locations of reported conflicts
- **License**: [ACLED Terms of Use](https://acleddata.com/terms-of-use/)

**Access for World Bank Staff:**

Access to ACLED data is provided through a contract between the World Bank and ACLED. World Bank employees can extract data by registering for an API key.

For assistance with data access, contact: [datalab@worldbank.org](mailto:datalab@worldbank.org)

## Installation

```bash
pip install -e .
```

## Dependencies

**Core Dependencies:**
- `pandas` - Data manipulation
- `geopandas` - Geospatial operations
- `shapely` - Geometric operations
- `h3` - Hexagonal spatial indexing
- `matplotlib` - Static plotting
- `contextily` - Basemap integration

**Optional Dependencies:**
- `folium` - Interactive web maps
- `jupyter` - Notebook support

## Usage Examples

See the tutorial notebooks for detailed examples:
- [Example Notebook](example.ipynb) - Basic usage and workflows
- [Extraction Notebook](extraction.ipynb) - Data extraction from ACLED API
- [Visualization Notebook](visualizing-acled.ipynb) - Creating maps and visualizations

## Contributing

This package follows Python best practices:
- **src-layout** for clean separation of source code and tests
- **Modular design** with focused, single-purpose modules
- **Type hints** for better code documentation
- **Comprehensive docstrings** for all public functions

Contributions are welcome! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the [Mozilla Public License 2.0](../LICENSE).

# Dashboard Pricing Dashboard

> A zero-to-run guide for first-time users: install Python, set up a virtual environment, install libraries, load your data, and launch the Dash app locally.

## Content Table

1. [What's In This Repo](#1-whats-in-this-repo)
   - [1.1 Core Modules](#11-core-modules)
   - [1.2 Key Libraries](#12-key-libraries)

2. [Installation Guide](#2-installation-guide)
   - [2.1 Install Python](#21-install-python)
   - [2.2 Clone from GitHub](#22-clone-from-github)
   - [2.3 Set up Virtual Environment & Install Libraries](#23-set-up-virtual-environment--install-necessary-libraries)

3. [Data Requirements](#3-data-requirements)
   - [3.1 External Data Requirements](#31-external-data-requirements)
     - [3.1.1 Pricing Dataset](#311-pricing-dataset)
     - [3.1.2 Product Categorization](#312-product-categorization)
   - [3.2 Internal Amazon Workflow](#32-internal-amazon-workflow)
     - [3.2.1 Repo Structure Review](#321-repo-structure-review)
     - [3.2.2 Acquire Data ](#322-workflow)

4. [Setup & Run Locally](#4-setup--run-locally)

5. [`pricing.sql` Documentation (Amazon Internal)](#5-sql-documentation-amazon-internal)
   - [5.1 Script Function Summary](#51-script-function-summary)
   - [5.2 SQL Script Documentation](#52-sql-script-documentation)


## 1. What’s In This Repo

### 1.1 Core modules the app imports:

```
app.py
built_in_logic.py
overview.py
opps.py
faq.py
navbar.py
home.py
```

### 1.2 Key libraries used:

- `dash`, `dash_bootstrap_components`, `dash_bootstrap_templates`
- `pandas`, `numpy`, `scipy`, `scikit-learn`
- `plotly`, `pygam`


## 2. Installation Guide


### 2.1: Install Python

Install **Python 3.11** (recommended):
- **Windows**: install from [python.org](https://www.python.org) or Microsoft Store
- **macOS**: `brew install python@3.11` or download from [python.org](https://www.python.org)


### 2.2: Clone from GitHub from terminal

```bash
git clone https://github.com/angie-xiao/pricing.git
```

### 2.3: Set up virtual environment & install necessary libraries

Windows
```bash
py -3.11 -m venv .pricing-venv
.\.pricing-venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install `
  "dash>=2.17" `
  "dash-bootstrap-components>=1.6" `
  "dash-bootstrap-templates>=1.2" `
  "pandas>=2.1" `
  "numpy>=1.26,<2.1" `
  "scipy>=1.11" `
  "scikit-learn>=1.3" `
  "pygam>=0.9.0,<1.0" `
  "plotly>=5.20"
```

macOS / Linux
```bash

python3.11 -m venv .pricing-venv
source .pricing-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  "dash>=2.17" \
  "dash-bootstrap-components>=1.6" \
  "dash-bootstrap-templates>=1.2" \
  "pandas>=2.1" \
  "numpy>=1.26,<2.1" \
  "scipy>=1.11" \
  "scikit-learn>=1.3" \
  "pygam>=0.9.0,<1.0" \
  "plotly>=5.20"
```



## 3. Data Requirements

This app supports users to provide their own data to generate insights. 
- For external stakeholders, please refer to Section 3.1 to make sure that your data meets the right format
- For Amazon internals, please refer to Section 3.2

### 3.1 External Data Requirements
Table must include following columns (including one row of dummy data)

#### 3.1.2 Pricing dataset: 

  order_date  | asin        | item_name          | shipped_units | revenue_share_amt | asp    | event_name
  ------------|-------------|--------------------|--------------|--------------------|--------|------------
  2025-07-01  | B00XYZABC1 | Brand X Dog Food   | 100         | 1299.99           | 12.99  | PD

#### 3.1.2 Product categorization:

  asin        | tag        | variation
  ------------|------------|------------
  B00XYZABC1 | Unscented  | 16lb


### 3.2 Internal (Amazon) workflow:

#### 3.2.1 Repo Structure Review

``` bash
pricing/
├── dash/                       # Dash app code
├── data/                       # << data storage folder
├── pricing.sql                 # << main data extraction script
```
#### 3.2.2 Acquire Data 

Run `pricing.sql` in [workbench](https://datacentral.a2z.com/workbench). 

Adjust filters (e.g. for marketplace, time window etc.) as needed.

```
pricing/
├── dash/
├── data/
│   ├── pricing.csv
│   ├── product_categorization.csv
```
 

## 4. Setup & Run locally

Remember to **ALWAYS** first activate your virtual environment before calling the main `app.py` script.

Windows
```bash
.\.pricing-venv\Scripts\Activate.ps1
cd dash
python app.py
# open http://127.0.0.1:8050 in your browser
# stop with CTRL+C
```

macOS / Linux
```bash
source .pricing-venv/bin/activate
cd dash
python app.py
# open http://127.0.0.1:8050 in your browser
# stop with CTRL+C
```

That's it! You can now access the visual results of this Dash app at http://127.0.0.1:8050 on your browser. 🎉


## 5. `pricing.sql` documentation (Amazon Internal)

### 5.1 Script functoin summary
The script enables analysis of:
- Deal performance by event type
- Discount depth impact
- Sales lift during promotions
- Revenue impact
- Promotional strategy effectiveness

### 5.2 SQL Script Documentation
The `pricing.sql` script analyzes promotional deal performance through the following steps:

1. **Base Promotion Information** (`base_promos`)
   - Filters US marketplace promotions (region_id = 1, marketplace_key = 7)
   - Captures promotions active in July 2025
   - Includes only Approved/Scheduled deals
   - Types: Best Deal, Deal of the Day, Lightning Deal, Event Deal
   - Excludes OIH promotions

2. **Event Categorization** (`raw_events`)
   - Classifies promotions into event tiers:
     * Tier 1: Prime Day, BSS, PBDD, T5/11/12
     * Tier 1.5: Back to School, Back to University
     * Tier 2-3: Black Friday, Cyber Monday, Boxing Week
     * Other: Uncategorized promotions

3. **Promotion Prioritization** (`promotion_details`)
   - Handles multiple promotions per ASIN
   - Priority order:
     1. Prime Day, BSS, PBDD, T5/11/12
     2. Black Friday
     3. Cyber Monday
     4. Boxing Week
     5. NYNY
     99. Other events

4. **Baseline Price Calculation** (`t4w`)
   - Calculates 4-week average selling price before promotion
   - Window: 28 days before to 1 day before promotion
   - Filters:
     * Retail merchant sales
     * New condition items
     * Shipped units only

5. **Performance Measurement** (`all_shipments`)
   - Captures July 2025 performance metrics
   - Aggregates:
     * Total shipped units
     * Total revenue

6. **Final Output** (`final_output`)
   - Combines all previous steps
   - Key metrics:
     * Product identification (ASIN, item name)
     * Pricing (deal price, baseline price)
     * Performance (shipped units, revenue)
     * Event categorization
     * Discount amount

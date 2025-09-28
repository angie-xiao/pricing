# Optima - Pricing Dashboard

> A zero-to-run guide for first-time users: install Python, set up a virtual environment, install libraries, load your data, and launch the Dash app locally.

## Content Table
- [Optima - Pricing Dashboard](#optima---pricing-dashboard)
  - [Content Table](#content-table)
  - [1. What’s In This Repo](#1-whats-in-this-repo)
    - [1.1 Core modules the app imports](#11-core-modules-the-app-imports)
    - [1.2 Key libraries used](#12-key-libraries-used)
  - [2. Data Requirements](#2-data-requirements)
    - [2.1 External Data Requirements](#21-external-data-requirements)
      - [2.1.1 Pricing dataset:](#211-pricing-dataset)
      - [2.1.1 Product categorization:](#211-product-categorization)
    - [2.2 Internal (Amazon) workflow:](#22-internal-amazon-workflow)
      - [2.2.1 Repo Structure Review](#221-repo-structure-review)
      - [2.2.2 Acquire Data](#222-acquire-data)
  - [3. Installation Guide](#3-installation-guide)
    - [3.1 Install Python](#31-install-python)
    - [3.2 Clone from GitHub from terminal](#32-clone-from-github-from-terminal)
    - [3.3 Set up virtual environment \& install necessary libraries](#33-set-up-virtual-environment--install-necessary-libraries)
    - [3.4 Setup \& Run locally](#34-setup--run-locally)
  - [4. `pricing.sql` documentation (Amazon Internal)](#4-pricingsql-documentation-amazon-internal)
    - [4.1 Script functoin summary](#41-script-functoin-summary)
    - [4.2 SQL Script Documentation](#42-sql-script-documentation)
      - [4.2.1. Base Promotion Information](#421-base-promotion-information)
      - [4.2.2 Deal Categorization Logic](#422-deal-categorization-logic)
      - [4.2.3 Promotion Processing](#423-promotion-processing)
      - [4.2.4 Final Output Assembly](#424-final-output-assembly)
  - [5. Technical Documentation](#5-technical-documentation)
    - [5.1 Data Engineering Pipeline](#51-data-engineering-pipeline)
      - [5.1.1 ETL](#511-etl)
      - [5.1.2 Weighting \& Robustness](#512-weighting--robustness)
    - [5.2 Train/Test Split](#52-traintest-split)
    - [5.3 Modeling](#53-modeling)
    - [5.4. Tuning](#54-tuning)
    - [5.5 Validation (RMSE made business friendly)](#55-validation-rmse-made-business-friendly)
    - [5.6 Quick references for Classes \& Functions](#56-quick-references-for-classes--functions)

---

## 1. What’s In This Repo

### 1.1 Core modules the app imports

```
app.py
built_in_logic.py
overview.py
opps.py
faq.py
navbar.py
home.py
```

### 1.2 Key libraries used

- `dash`, `dash_bootstrap_components`, `dash_bootstrap_templates`
- `pandas`, `numpy`, `scipy`, `scikit-learn`
- `plotly`, `pygam`


## 2. Data Requirements

This app supports users providing their own data to generate insights.
- For external users, see 2.1 to format your data.
- For Amazon internal users, see 2.2.

### 2.1 External Data Requirements
Table must include following columns (including one row of dummy data)

#### 2.1.1 Pricing dataset: 

  order_date  | asin        | item_name          | shipped_units | revenue_share_amt | asp    | event_name
  ------------|-------------|--------------------|--------------|--------------------|--------|------------
  2025-07-01   | B00ABCD | Best Dog Food   | 100         | 1299.99           | 12.99  | PD

#### 2.1.1 Product categorization:

  asin        | tag        | variation
  ------------|------------|------------
  B00ABCD | Adult Dog  | 16lb


### 2.2 Internal (Amazon) workflow:

#### 2.2.1 Repo Structure Review

``` bash
pricing/
├── dash/                       # Dash app code
├── data/                       # << data storage folder
├── pricing.sql                 # << main data extraction script
```
#### 2.2.2 Acquire Data 

Run `pricing.sql` in [workbench](https://datacentral.a2z.com/workbench). 

Adjust filters (e.g. for marketplace, time window etc.) as needed.

```
pricing/
├── dash/
├── data/
│   ├── pricing.csv
│   ├── product_categorization.csv
```
 
 For the full description, see Section 4, [`pricing.sql` documentation (Amazon Internal)](#5-pricingsql-documentation-amazon-internal).

## 3. Installation Guide

### 3.1 Install Python

Install **Python 3.11** (recommended):
- **Windows**: install from [python.org](https://www.python.org)
  - [Amazon Internal] Search for Python & download from **Software Center**
- **macOS**: `brew install python@3.11` or download from [python.org](https://www.python.org)


### 3.2 Clone from GitHub from terminal

```bash
git clone https://github.com/angie-xiao/pricing.git
```

### 3.3 Set up virtual environment & install necessary libraries

Windows
```bash
# create and activate virtual environment
py -3.11 -m venv .pricing-venv           # only run once
.\.pricing-venv\Scripts\Activate.ps1     # acitvate

# install up-to-date pip
python -m pip install --upgrade pip

# keep build tools fresh
python -m pip install -U pip setuptools wheel

# install your requirements (will upgrade scikit-learn per the file)
pip install -U -r requirements.txt
```

macOS / Linux
```bash
# create and activate virtual environment
python3.11 -m venv .pricing-venv    # only run once
source .pricing-venv/bin/activate   # activate

# install up-to-date pip
python -m pip install --upgrade pip

# keep build tools fresh
python -m pip install -U pip setuptools wheel

# install your requirements (will upgrade scikit-learn per the file)
pip install -U -r requirements.txt
```


### 3.4 Setup & Run locally

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


## 4. `pricing.sql` documentation (Amazon Internal)

### 4.1 Script functoin summary

  Aside from supporting built-in logic (i.e., data engineering, modeling, tuning, visualizing etc.) of this project, data extracted by this script could also enable analysis of:
  - Deal performance by event type
  - Discount depth impact
  - Revenue impact
  - Promotional strategy effectiveness

For expected output table format, please refer to Section [4.2.4 Final Output Assembly](#424-final-output-assembly)

### 4.2 SQL Script Documentation

The `pricing.sql` script analyzes promotional deal performance through the following steps:

#### 4.2.1. Base Promotion Information
   - Captures promotions based on user definition, e.g. -
     - region_id = 1
     - marketplace = 7
     - GL = 199
     - time window = last 2 years
   - Includes only "Approved" / "Scheduled" deals
   - Types: Best Deal, Deal of the Day, Lightning Deal, Event Deal
   - Excludes OIH promotions

#### 4.2.2 Deal Categorization Logic

  The script implements a sophisticated deal categorization system:

  1. **Standardized Deal Periods**
     - Identifies common start/end dates for major events (Prime Day, Black Friday, etc.)
     - Requires minimum 3 deals with matching patterns to establish standard dates
     - Ensures consistent reporting and analysis

  2. **Flexible Event Recognition**
     - Accommodates floating date events (e.g., Prime Day in late June or July)
     - Pattern matching in promotion titles:
       - Direct event names (e.g., "Prime Day")
       - Known acronyms (e.g., "PD" for Prime Day)
       - Date-based validation for seasonal events

  3. **Event Name Tagging**
     - Based on `promotion_internal_title` field
     - Event types:
       - HVE (High Velocity Events)
          - Tier 1: Prime Day, BSS, PBDD, T5/11/12
          - Tier 1.5: Back to School, Back to University
       - Other (Tier 2 +)
         - Defaults to "OTHER" if no matching pattern
      - Excludes promotions with "OIH" in title

  4. **Overlap Resolution**
     - Prioritizes higher tier events when dates overlap
     - Example priority order:
        - For HVEs, Tier 1 events get the highest priority
          - ... followed by Tier 1.5, Tier 2, Tier 3 etc.
        - Regular promotions


#### 4.2.3 Promotion Processing
1. **Base Promotion Extraction**
   - Captures approved/scheduled promotions
   - Types: Best Deal, Deal of the Day, Lightning Deal, Event Deal
   - Excludes OIH promotions

2. **Event Categorization**
   - Identifies major events (Prime Day, BSS, etc.)
   - Uses both date logic and title pattern matching
   - Priority order implemented for overlapping events

3. **Date Standardization**
   - Identifies common start/end dates for events
   - Requires minimum 3 promotions for pattern recognition
   - Consolidates promotion dates for consistency

#### 4.2.4 Final Output Assembly
- Transaction Classification Logic:
  * **Critical Price Validation**: Orders are classified as BAU (Business As Usual) if:
    - Customer paid non-deal price during events (e.g., non-Prime members during Prime Day)
    - ⚠️ **Edge Case**: Actual purchase price doesn't match promotional price
      - e.g., if a non-Prime customer made a purchase during Prime Day and pays for non-deal price, this is considered as a **BAU** purchase
    - No active promotion exists for the ASIN
  * This ensures accurate event attribution based on actual customer purchase behavior

- Key Metrics Output:
  * ASIN
  * Item name
  * Order date
  * GL product group
  * Company information
  * Price
  * Event classification
  * Promotion pricing details
  * Aggregated units and revenue

- Units and revenue are summed by
  - ASIN
  - Order date
  - Price
  - Event type (BAU vs HVE)

    
## 5. Technical Documentation

This section explains, in order, how raw rows become the curves and KPIs you see in the Dash app. The flow is:

> (1) Data wrangling → (2) Train/Test split → (3) Modeling → (4) Tuning → (5) Validation & RMSE → (6) Visualization through Dash

### 5.1 Data Engineering Pipeline

- **Goal:**
  - turn raw transactions & product tags into a tidy, model-ready table.

- **Inputs:**
  - `data/pricing.csv`: asin, order_date, price, shipped_units, event_name, …
  - `data/product_categorization.csv`: asin → product label/tags

#### 5.1.1 ETL
- **Process:**
  1. **Normalize & merge:**
     - Clean column names, merge `pricing.csv` with product `tags` on `asin`;
     - Create unified  `product` label used for grouping.

  2. **Type safety + non-negativity:**
     - Coerce `order_date` to datetime; 
     - Coerce `price`/`shipped_units` to numeric and clip at 0.

  3. **Price-day coverage:**
     - Compute `days_sold` per (`asin`, `price`);
     - Merge back so each row knows how long that ASP was live.

  4. **Aggregate on daily granularity:**
     - Build a daily aggregate (by `asin`, `event_name`, `order_date`) to compute an `ASP` and daily metrics;
     - Then join back to enrich the row grain.
     - The benefit of doing this is edge casing scenarios where actual purchase price doesn't match promotional price (see Section [4.2.4 Final Output Assembly](#424-final-output-assembly) for more details).

  
  5. **Top-N focus:**
     - Rank products by total `revenue` and keep only the top-N bestselling products. 
      - Parameter `top_n`: configurable (default 10) - for modeling focus and speed.

  6. **Categorical variables encoding:**
     - Label-encode categorical variables for model inputs
       - Target columns
         - `event_name` → `event_encoded` and
         - `product` → `product_encoded` 
      - e.g., BAU = 0, PD = 1, PBDD = 2, ... etc.

  7. **Output: a tidy frame with:**
    - `product, asin, order_date, price, shipped_units, event_encoded, product_encoded, days_sold, (optional) deal_discount_percent`


#### 5.1.2 Weighting & Robustness

  Each observation’s weight blends three signals, then normalizes:

  1.  **Recency (time-decay)**

      - Newer rows count more:
      $$𝑤_{time} = exp*({decay\:rate} * days\:apart)$$

      - `decay_rate` is tuned 
      - typical range ≈ –0.05…–0.001
    
  2. **Rarity (density-aware, bounded)**
 
       - Rare ASPs get a small boost to better cover the price range: 
  
       $$𝑤_{rarity} ∝ (1/density)^{𝛽}$$

       - Uses a smoothed histogram of prices to compute a rarity multiplier
         -  capped (e.g., ≤ 1.25).
       - Only the tails (outside, say, 10th–90th percentiles) get this bonus to avoid double-counting the core.

  3.  **Local leverage cap**

        - A rolling window cap keeps any single point from dominating:

      $$𝑤_{i} ≤ l_{cap} * local\: mean (𝑤) \: sorted \: by \: price $$


  4. **(Optional) Tail boost**
        - Distance from median price; 
        - Disabled by default to avoid double-counting rarity.

  5. **Robust residual reweighting (OOF Huber)**
     - After a quick out-of-fold fit, high-residual rows are softly down-weighted (Huber).
     - This step keeps odd spikes from steering the curve while preserving valid signal.
  

### 5.2 Train/Test Split
- **Goal:** 
  - Evaluate the model on **unseen** data for each product.

- **How we split:**
  - **Primary:** time-based split using a quantile cut of `order_date` (e.g., last ~20% for **test**)
  - **Fallback:** deterministic random split when dates are too sparse
  - Ensures both train & test have enough rows (guarded minimums)

- **What gets returned**
  - `X_train`, `y_train`, `w_train` for fit; `X_all` for prediction over the full price grid.
  - `split` column on the per-row frame marks `train` vs `test`.


### 5.3 Modeling

- **Goal:** 
  - Estimate the units-vs-price curve per product (with event effects).

- **Model:** 
  - **Expectile GAM** per product (e.g., at q=0.025, 0.5, & 0.975)
    - `s(price)` (adaptive spline, #splines tuned)
    - `s(deal_discount_percent)` (moderate spline)
    - `f(event_encoded)` (categorical factor)
  - Why expectiles? 
    - It lets us fit a median curve and upper/lower bands without assuming Gaussian noise. 
  - Fitted with the engineered `sample_weight` from 6.1.3.
- **Stability guardrails:**
  - **Edge tapers**: gently shrink predictions beyond the train-price anchors
(weighted quantiles, e.g., 10%/90%), with configurable stretches.
  - **Non-crossing bands**: enforce $q_{low} ≤ q_{med} ≤ q_{high}$

- **Optional ensemble**
  - **Bootstrap** with early stop (targeting a relative SE) to stablize bands.

 
### 5.4. Tuning

- **Weight Tuner**:
  - Instead of a massive grid, a fast **random sampling + successive-halving** tuner searches over:
    - `decay_rate, rarity_beta, rarity_cap, tail_q, lcap `
    - (plus `tail_strength` &`tail_p` if tail boost is enabled).
  - Cheap proxy scoring (constant/linear fits + dispersion penalties) → finalists → full-data check.
  - The best set is cached and re-used automatically by `_make_weights()`.

- **GAM Tuner (inside `GAMTuner`)**
  - Coarse search over `n_splines(price)` * `λ` (regularization) -> fine search near the best `λ`
  

### 5.5 Validation (RMSE made business friendly)
- Raw metric 
  
    $$RMSE_{units} = \sqrt{\frac{1}{N_{test}} * ∑(y_{actual} - y_{pred})^2} $$

- How we present it
  - Per product (in cards/tooltips):
    - “Typical error ≈ X units/day (held-out).”
     - Optional % error proxy: $RMSE / max(1, median\: daily\: units)$
  - Portfolio (Top-N)
    - Report the **median** per-product RMSE (robust to outliers)
    - Optionaly an **inventory-weighted** roll-up
  - Dollar view (optional)
    - * Multiply unit RMSE by a representative  ASP to estimate "~$ impact / day" - clearly labeled as approximate

### 5.6 Quick references for Classes & Functions 

* **Data engineering (ETL + weights)**
  * `DataEngineer.prepare()`
  * `_make_weights() `
  * ... and helpers

* **Split**
  * `_make_design_matrix() ` - adds split, returns train/test matrices

* **Modeling**
  * `GAMTuner.fit()`
  * `GAMModeler._fit_expectiles()`

* **Guardrails**
  * `GAMModeler._apply_edge_tapers()` + non-crossing

* **Tuning**
  * `DataEngineer._grid_search_weights(); `
  * GAMTuner for GAM smoothness

* **Validation**
  * test-set RMSE computed on rows with split == "test"; 
  * displayed in cards/roll-ups
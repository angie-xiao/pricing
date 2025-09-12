# Dashboard Pricing Dashboard

> A zero-to-run guide for first-time users: install Python, set up a virtual environment, install libraries, load your data, and launch the Dash app locally.

---

## Content Table

  - [0. What’s in this repo](#whats-in-this-repo)
  - [1. Installation Guide](#installation-guide)
    - [Step 1: Install Python](##step-1-install-python)
    - [Step 2: Clone from GitHub](##step-2-clone-from-github)
    - [Step 3: Set up virtual environment \& install necessary libraries](##step-3-set-up-virtual-environment--install-necessary-libraries)
    - [Step 4: Grab your data](##step-4-grab-your-data)
      - [Pricing dataset (required):](###pricing-dataset-required)
      - [Product categorization (required):](#product-categorization-required)
      - [Internal (Amazon) workflow:](#internal-amazon-workflow)
    - [Step 5: Put data in the right folder](#step-5-put-data-in-the-right-folder)
        - [Repo structure (reference)](#repo-structure-reference)

    - [Step 6: Run locally](#step-6-run-locally)

---

## What’s in this repo

Core modules the app imports:

```
app.py
built_in_logic.py
overview.py
opps.py
faq.py
navbar.py
home.py
```

Key libraries used:

- `dash`, `dash_bootstrap_components`, `dash_bootstrap_templates`
- `pandas`, `numpy`, `scipy`, `scikit-learn`
- `plotly`, `pygam`

---

# Installation Guide


## Step 1: Install Python

Install **Python 3.11** (recommended):
- **Windows**: install from [python.org](https://www.python.org) or Microsoft Store
- **macOS**: `brew install python@3.11` or download from [python.org](https://www.python.org)

---

## Step 2: Clone from GitHub from terminal

```bash
git clone https://github.com/angie-xiao/pricing.git
```
---

## Step 3: Set up virtual environment & install necessary libraries

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

---

## Step 4: Grab your data

### 4.1 Requirements
#### 4.1.1 Pricing dataset (required):
* Must include the following fields:
  - order_date
  - asin
  - item_name
  - shipped_units
  - revenue_share_amt
  - asp
  - event_name (e.g. PBDD/ PD/ BAU etc.)

#### 4.1.2 Product categorization (required):
* Must include:
  - asin
  - tag (product name; e.g. Unscented)
  - variation (e.g. 16lb)

### 4.2 Internal (Amazon) workflow:
* Run the `pricing.sql` script provided
* Use Turismo or manual download to extract product tags / variation info
* Categorize if needed

---

## Step 5: Put data in the right folder

Make sure your input files are placed under the `data/` folder at the project root.

```
pricing/
├── dash/
├── data/
│   ├── pricing.csv
│   ├── product_categorization.csv
```

If your file names differ, update the file paths in `dash/built_in_logic.py`, `overview.py`, or `opps.py`.


### Repo structure (reference)
``` bash
pricing/
├── .pricing-venv/              # local virtual environment
├── .vscode/                    # editor settings
├── dash/                       # Dash app code
│   ├── app.py
│   ├── built_in_logic.py
│   ├── overview.py
│   ├── opps.py
│   ├── faq.py
│   ├── navbar.py
│   └── home.py
├── data/                       # <<< put pricing & product categorization files here
│   ├── pricing.csv
│   └── product_categorization.csv
├── pricing.sql                 # SQL script for Amazon internal use
├── .gitignore
└── README.md
```

---

## Step 6: Run locally

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

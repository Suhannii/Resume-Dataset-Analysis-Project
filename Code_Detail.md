# Experiment 1: Data Collection and Preprocessing Techniques

## Overview

This project performs data preprocessing on a real-world resume dataset (`resume_dataset_200k_enhanced.csv`) containing **200,000 candidate records**. The notebook walks through every essential preprocessing step required before feeding data into any machine learning model.

---

## Dataset

| Property | Details |
|----------|---------|
| File | `resume_dataset_200k_enhanced.csv` |
| Rows | 200,000 |
| Columns | 17 |
| Target Column | `hired` (0 = Not Hired, 1 = Hired) |

### Columns Description

| Column | Type | Description |
|--------|------|-------------|
| `candidate_id` | int | Unique ID for each candidate |
| `age` | int | Age of the candidate |
| `education_level` | object | Highest education (Bachelors / Masters / PhD) |
| `university_tier` | object | University ranking (Tier 1 / Tier 2 / Tier 3) |
| `cgpa` | float | Cumulative Grade Point Average (0–10) |
| `internships` | int | Number of internships completed |
| `projects` | int | Number of projects done |
| `programming_languages` | int | Number of programming languages known |
| `certifications` | int | Number of certifications earned |
| `experience_years` | float | Total years of work experience |
| `hackathons` | int | Number of hackathons participated in |
| `research_papers` | int | Number of research papers published |
| `skills_score` | float | Technical skills score |
| `hired` | int | Target variable — 1 if hired, 0 if not |
| `soft_skills_score` | float | Soft skills score |
| `resume_length_words` | int | Word count of the resume |
| `company_type` | object | Type of company (MNC / Startup / etc.) |

---

## Requirements

Install all required libraries before running the notebook:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## Notebook Structure

### 1. Import Libraries

Imports all necessary Python libraries:

- `pandas` — data loading, manipulation, and analysis
- `numpy` — numerical operations
- `matplotlib` & `seaborn` — data visualization
- `sklearn.preprocessing` — scaling and encoding tools

Also sets display options so all columns are visible and floats are formatted to 4 decimal places.

---

### Step 1 — Load the Dataset

```python
data = pd.read_csv('resume_dataset_200k_enhanced.csv')
```

Reads the CSV file from disk into a pandas DataFrame. Prints the shape `(200000, 17)` to confirm successful loading.

---

### Step 2 — Inspect the Dataset

Three sub-steps to understand the data before touching it:

- `data.head()` — shows the first 5 rows to get a feel for the data
- `data.info()` — shows column names, data types, and non-null counts
- `data.describe()` — shows statistical summary (mean, std, min, max, quartiles) for all numerical columns

This step helps identify which columns are numerical vs categorical, and whether any values look suspicious.

---

### Step 3 — Handle Missing Values (Deletion)

**Why:** Machine learning models cannot work with NaN/null values. They must be handled before any further processing.

**What the code does:**

```python
data.isnull().sum()
```
Counts missing values in every column. If a column shows 0, it has no missing data.

```python
data_cleaned = data.dropna()
```
Drops every row that contains at least one missing value. The result is stored in `data_cleaned` so the original `data` is preserved for reference.

> In this dataset, there are no missing values, so `data_cleaned` will have the same shape as `data`. This step is still important to include as a standard practice.

---

### Step 4 — Normalize Data (Min-Max Scaling)

**Why:** Features like `age` (18–65) and `skills_score` (0–30) are on very different scales. Normalization brings them all to the same range so no single feature dominates the model.

**Formula:**
```
x_norm = (x - min) / (max - min)
```

**Result:** Every value in the selected columns will be between **0 and 1**.

**Columns scaled:**
`age`, `cgpa`, `experience_years`, `skills_score`, `soft_skills_score`, `resume_length_words`

```python
scaler_mm = MinMaxScaler()
data_normalized[scale_cols] = scaler_mm.fit_transform(data_cleaned[scale_cols])
```

The result is stored in `data_normalized`. The original `data_cleaned` is not modified.

---

### Step 5 — Standardize Data (Z-score / StandardScaler)

**Why:** Some algorithms (like SVM, Logistic Regression, PCA) assume features are normally distributed with mean 0 and standard deviation 1. Standardization achieves this.

**Formula:**
```
x_std = (x - mean) / std
```

**Result:** Every scaled column will have **mean ≈ 0** and **std ≈ 1**.

```python
scaler_std = StandardScaler()
data_standardized[scale_cols] = scaler_std.fit_transform(data_cleaned[scale_cols])
```

The result is stored in `data_standardized`. A `.describe()` check confirms mean and std values are correct.

> Difference from normalization: Min-Max keeps values in [0,1] but is sensitive to outliers. Standardization is more robust to outliers but does not bound values to a fixed range.

---

### Step 6 — Encode Categorical Variables: One-Hot Encoding

**Why:** Machine learning models work with numbers, not text. Categorical columns like `education_level` must be converted to numeric form.

**One-Hot Encoding** creates a new binary column for each unique category value.

Example — `education_level` with values `Bachelors`, `Masters`, `PhD` becomes:

| education_level_Bachelors | education_level_Masters | education_level_PhD |
|--------------------------|------------------------|---------------------|
| 1 | 0 | 0 |
| 0 | 1 | 0 |

```python
data_ohe = pd.get_dummies(data_ohe, columns=['education_level', 'university_tier', 'company_type'])
```

**Best used for:** Nominal categories where there is no ranking or order between values.

---

### Step 7 — Encode Categorical Variables: Label Encoding

**Why:** An alternative to One-Hot Encoding that assigns a single integer to each category instead of creating multiple columns.

Example — `university_tier` becomes:
- `Tier 1` → 0
- `Tier 2` → 1
- `Tier 3` → 2

```python
le = LabelEncoder()
data_le[f'{col}_encoded'] = le.fit_transform(data_le[col])
```

The mapping for each column is printed so you can see exactly which number maps to which category.

**Best used for:** Ordinal categories where there is a natural ranking (e.g., Tier 1 is better than Tier 3).

> Difference from One-Hot: Label Encoding uses 1 column per feature (compact) but implies an order. One-Hot uses multiple columns but makes no assumption about order.

---

### Step 8 — Visualize Before & After Preprocessing

**Why:** Visualization confirms that preprocessing steps worked correctly and helps communicate the effect to others.

**Plot 1 — Histogram comparison (3 panels):**
- Original `experience_years` distribution
- After Min-Max Normalization (values compressed to [0,1])
- After Standardization (values centered around 0)

**Plot 2 — Box Plot comparison (2 panels):**
- `cgpa`, `skills_score`, `soft_skills_score` before scaling
- Same columns after Min-Max Normalization

Box plots clearly show how all features are brought to the same scale after normalization, making them directly comparable.

---

## Summary Table

| Step | Technique | Input | Output |
|------|-----------|-------|--------|
| Step 3 | `dropna()` | Raw data with possible nulls | `data_cleaned` — null-free |
| Step 4 | Min-Max Scaler | `data_cleaned` | `data_normalized` — values in [0,1] |
| Step 5 | Standard Scaler | `data_cleaned` | `data_standardized` — mean=0, std=1 |
| Step 6 | One-Hot Encoding | `data_cleaned` | `data_ohe` — binary columns per category |
| Step 7 | Label Encoding | `data_cleaned` | `data_le` — integer per category |
| Step 8 | Histograms & Box Plots | All above | Visual comparison |

---

## How to Run

1. Make sure `resume_dataset_200k_enhanced.csv` is in the same folder as the notebook.
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Open `resumeAnalysis.ipynb` in Jupyter Notebook or VS Code.
4. Run all cells top to bottom (`Kernel > Restart & Run All`).

---

## Key Takeaways

- Always inspect your data before processing — know your column types, ranges, and nulls.
- Use **Min-Max Normalization** when you need values in a fixed range [0,1].
- Use **Standardization** when your algorithm assumes normally distributed features.
- Use **One-Hot Encoding** for nominal categories (no order).
- Use **Label Encoding** for ordinal categories (has order).
- Always visualize before and after to verify your preprocessing worked as expected.

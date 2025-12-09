# Solar Irradiation Forecast Dashboard - Complete Documentation

## Table of Contents
1. [What This Program Does](#what-this-program-does)
2. [The Big Picture](#the-big-picture)
3. [How to Use It](#how-to-use-it)
4. [Step-by-Step: What Happens Inside](#step-by-step-what-happens-inside)
5. [Understanding the Results](#understanding-the-results)
6. [Troubleshooting](#troubleshooting)
7. [Glossary of Terms](#glossary-of-terms)

---

## What This Program Does

**In simple terms:** This program learns from your past solar energy measurements and weather data to predict how much solar energy will be available in the future—3 hours from now, 24 hours from now, and 7 days from now.

Think of it like a weather forecaster, but instead of predicting rain, it predicts solar irradiance (the energy from the sun hitting the Earth).

**Why is this useful?**
- Solar farms need to know how much power they'll generate to manage the grid
- Businesses can plan energy usage based on expected solar availability
- Battery storage systems can prepare for sunny or cloudy periods

---

## The Big Picture

### What You Give It (Input)
You upload a CSV file (a spreadsheet saved as plain text) containing:
- **Date and time information** (Year, Month, Day, Hour, Minute)
- **Past solar measurements** (GHI - the main measurement)
- **Weather data** (Temperature, Humidity, Pressure, Wind Speed, etc.)

### What It Does (Process)
1. Reads your historical data
2. Creates new calculated features from the data
3. Splits the data into "learning" and "testing" portions
4. Trains three separate machine learning models (one for each time horizon)
5. Evaluates how accurate each model is

### What You Get Back (Output)
- **Performance scores** showing how accurate the predictions are
- **Detailed charts** showing actual vs predicted values
- **Feature importance** explaining which factors matter most
- **Error analysis** showing where predictions go wrong
- **Downloadable file** with all predictions you can use elsewhere

---

## How to Use It

### Installation (First Time Setup)

1. **Open PowerShell** (press `Windows + R`, type `powershell`, press Enter)

2. **Navigate to your folder:**
   ```powershell
   cd C:\Users\Darjay\.qodo\solar_dashboard
   ```

3. **Create a virtual environment** (keeps dependencies organized):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

4. **Install required packages:**
   ```powershell
   pip install -r requirements.txt
   ```

### Running the Program

1. **Start the app:**
   ```powershell
   streamlit run app.py
   ```

2. **A window opens** automatically in your browser (usually `http://localhost:8501`)

3. **You see the app interface** with an upload button

### Using the App

1. **Click "Upload a CSV file"**

2. **Select your data file** (must contain Year, Month, Day, Hour, Minute and GHI columns)

3. **Wait for it to process** (you'll see progress messages)

4. **Review the results:**
   - See how many rows of data you have
   - Check the column names detected
   - View a preview of your data
   - See performance metrics
   - Study the charts
   - Download your predictions

---

## Step-by-Step: What Happens Inside

### Step 1: Reading Your File

**What happens:**
- The program reads your CSV file
- It shows you how many rows (records) you uploaded
- It lists all the column names it detected

**What you see:** 
```
Loaded 35040 rows.
Detected columns: Year, Month, Day, Hour, Minute, Temperature, GHI, ...
```

---

### Step 2: Fixing Column Names

**What happens:**
Some data files use shortened or different names for columns. The program recognizes common variations:

| Short Name | Standard Name |
|-----------|---------------|
| "Relative H" | "Relative Humidity" |
| "Solar Zenit" | "Solar Zenith" |
| "Surface All" | "Surface Albedo" |
| "Clearsky D" | "Clearsky DHI" |
| "Clearsky G" | "Clearsky GHI" |
| "Clearsky D 2" | "Clearsky DNI" |

**Why?** Different weather stations save data with different naming conventions, and this makes the program flexible.

---

### Step 3: Converting to Numbers

**What happens:**
- The program checks every column
- It converts text that looks like numbers into actual numbers
- It replaces placeholder values like "N/A" or "NA" with blanks (missing values)

**Why?** Machine learning needs clean numbers to work with. A computer can't understand "N/A"—it needs blank spots or actual numbers.

**Example:**
```
Before: "25.5", "N/A", "23.1"
After:  25.5,   [blank], 23.1
```

---

### Step 4: Creating a Timeline

**What happens:**
Your separate Year, Month, Day, Hour, Minute columns are combined into a single date-time:

**Example:**
```
Year: 2025     |
Month: 12      | → Combined into: 2025-12-09 14:30
Day: 9         |
Hour: 14       |
Minute: 30     |
```

**Why?** This creates proper time ordering and allows the program to understand "what came before" and "what came after."

---

### Step 5: Feature Engineering (The Smart Part)

**What is "feature engineering"?** 
Creating new useful information from your raw data. Imagine a teacher learning from a student's past test scores to predict the next test—they look at recent scores, trends, patterns of improvement. The program does this automatically.

#### **What new features are created:**

##### **A. Lag Features (Recent History)**
These look at past values:

```
GHI_lag1   = GHI from 1 hour ago
GHI_lag2   = GHI from 2 hours ago
GHI_lag3   = GHI from 3 hours ago
GHI_lag6   = GHI from 6 hours ago
GHI_lag12  = GHI from 12 hours ago
GHI_lag24  = GHI from 24 hours ago
GHI_lag48  = GHI from 48 hours ago
```

**Why?** Solar energy is very predictable in the short term—if it's sunny now, it'll probably be sunny in 1 hour. The model learns these patterns.

##### **B. Weather Lags**
Same concept but for Temperature, Humidity, Pressure, etc.:

```
Temperature_lag1
Temperature_lag6
Temperature_lag12
Humidity_lag1
Humidity_lag6
... and so on
```

**Why?** Weather patterns develop slowly. A warm afternoon usually means morning temperatures give clues.

##### **C. Rolling Statistics (Trends)**
These calculate averages and changes over recent periods:

```
GHI_roll3_mean  = Average GHI over last 3 hours
GHI_roll3_std   = How much GHI varied in last 3 hours (volatility)
GHI_roll6_mean  = Average GHI over last 6 hours
GHI_roll6_std   = How much GHI varied in last 6 hours
GHI_roll24_mean = Average GHI over last 24 hours
GHI_roll24_std  = How much GHI varied in last 24 hours
```

**Why?** A trend matters. Is GHI increasing or decreasing? Is the weather stable or chaotic? These numbers capture that.

##### **D. Calendar Features (Time of Day/Season)**
These capture natural solar patterns:

```
hour      = 0 to 23 (solar energy is highest at noon)
dayofweek = 0 to 6 (which day of the week—not usually predictive for solar)
month     = 1 to 12 (solar energy is higher in summer, lower in winter)
```

**Why?** The sun's position changes throughout the day and year. The model learns "it's 2am so solar will be zero" and "it's July so expect more sun."

---

### Step 6: Cleaning Up Incomplete Rows

**What happens:**
The program removes rows that have missing values in important columns (the features it needs).

**Why?** Machine learning models can't learn from incomplete information. It's like trying to study for a test but some pages of the textbook are blank.

**Important:** 
- Rows where ANY feature is missing are removed
- But rows where only the TARGET (future GHI) is missing are kept—each forecast model will handle those separately

**Example:**
```
Row 1: Year, Month, Day, Hour, Minute, Temp, Humidity, GHI = KEEP (complete)
Row 2: Year, Month, Day, Hour, Minute, Temp, [blank], GHI = REMOVE (missing Humidity)
Row 3: Year, Month, Day, Hour, Minute, Temp, Humidity, [blank] = MAYBE (depends on horizon)
```

**Result:** You might start with 105,120 rows but end with 34,000 rows after cleaning.

---

### Step 7: Splitting into Train & Test Sets

**What happens:**
The program splits your clean data into two parts:
- **80% Training data** (used to teach the model)
- **20% Validation data** (used to test how well it learned)

**Why?** You can't test a student on the exact same material they studied—they might just memorize answers. You need to test them on new questions.

**Important:** The split is chronological—earlier data is used for training, newer data for testing. This is realistic because you're predicting the future.

---

### Step 8: Creating the Future Targets

**What happens:**
The program creates three new columns representing the future:

```
GHI_tplus3   = GHI value 3 hours in the future
GHI_tplus24  = GHI value 24 hours in the future
GHI_tplus168 = GHI value 7 days (168 hours) in the future
```

**How?** It shifts the GHI column forward:

```
Original time 2025-12-09 10:00: GHI = 500
New column GHI_tplus3 at 2025-12-09 10:00: GHI_tplus3 = 480
     (This is the actual GHI at 2025-12-09 13:00)
```

**Why?** The model learns to predict these future values based on current and past features.

---

### Step 9: Training Three Separate Models

**What is a machine learning model?**
A mathematical formula that learns patterns from data. It's like teaching a student to recognize cats by showing hundreds of cat pictures—eventually they learn "if it has pointy ears, whiskers, and meows, it's probably a cat."

**Why three models?**
Each time horizon (3h, 24h, 7d) has different patterns and challenges:
- **3-hour forecast:** Very predictable (weather doesn't change much in 3 hours)
- **24-hour forecast:** Moderate difficulty (weather can change a lot in a day)
- **7-day forecast:** Very hard (weather is chaotic over a week)

Each model gets trained separately so it can learn the best strategy for its specific time horizon.

#### **How training works:**

1. **Feed data to the model:** Show it thousands of examples: "Here's what the weather looked like, here's the solar energy 3 hours later"

2. **Model makes predictions:** At first, these are terrible and random

3. **Compare to reality:** "You predicted 500 W/m², but it was actually 480 W/m²"

4. **Model adjusts itself:** It tweaks its internal mathematical formulas to do better next time

5. **Repeat:** This happens 600 times (600 "rounds" or "trees" of learning)

**Result:** A trained model that can make educated guesses about future solar energy

---

### Step 10: Testing on New Data

**What happens:**
The program uses the 20% validation data (data the model has never seen) to test accuracy:

1. **Feed features to the model:** "Here are current conditions. Predict solar energy 3 hours from now."

2. **Get predictions:** The model makes its guess

3. **Compare to actual:** Check what actually happened

4. **Calculate errors:** 
   - **MAE (Mean Absolute Error):** Average error amount (e.g., off by 25 W/m² on average)
   - **RMSE (Root Mean Squared Error):** A stricter version that penalizes big mistakes more
   - **R² (R-squared):** A score from 0 to 1 (1 = perfect, 0 = useless). Tells you what fraction of the variation is explained.

---

### Step 11: Creating Visualizations (The Charts)

The program creates three main types of charts:

#### **Chart 1: Actual vs Predicted**
Shows two lines over time:
- **Blue line:** What actually happened (real solar measurements)
- **Orange line:** What the model predicted

**What to look for:**
- Lines close together = good predictions
- Lines far apart = poor predictions
- Patterns in the differences = model bias (consistently over/under predicting)

#### **Chart 2: Feature Importance**
Shows which factors matter most for predictions, ranked by importance:

**Example:**
```
GHI_lag1        ████████░ (80%) — Very important
Temperature_lag6 ████░░░░░ (40%) — Moderately important
Humidity_lag12   ██░░░░░░░ (20%) — Less important
month           █░░░░░░░░ (10%) — Least important
```

**Why?** Helps you understand: "What drives solar energy predictions?"

#### **Chart 3: Residual Analysis**
Shows prediction errors as dots on a scatter plot:

**What to look for:**
- **If dots form a random cloud:** Good! Errors are random (no bias)
- **If dots form a pattern:** Bad! The model has a systematic bias
  - Dots above the center line = model underestimated
  - Dots below = model overestimated

---

### Step 12: Generating Your Output File

**What happens:**
All original data plus the predictions are saved to `predictions.csv`

**What's included:**
```
timestamp, Year, Month, Day, Hour, Minute, Temperature, GHI, 
GHI_pred_tplus3, GHI_pred_tplus24, GHI_pred_tplus168
```

**What you can do with it:**
- Import into Excel or another tool
- Plot your own charts
- Share with colleagues
- Use for further analysis

---

## Understanding the Results

### The Metrics Table

You'll see a table like this:

| Metric | tplus3 (3h) | tplus24 (24h) | tplus168 (7d) |
|--------|-----------|---------------|--------------|
| MAE    | 25.3      | 45.2          | 78.5         |
| RMSE   | 35.1      | 62.4          | 105.3        |
| R²     | 0.92      | 0.78          | 0.45         |

#### **What these numbers mean:**

**MAE (Mean Absolute Error):**
- **Definition:** Average prediction error in W/m²
- **Interpretation:** 
  - 25.3 means "on average, we're off by 25.3 W/m² for the 3-hour forecast"
  - Lower is better
  - Compare to the typical range of GHI values to judge if it's good

**RMSE (Root Mean Squared Error):**
- **Definition:** Similar to MAE but punishes big errors more
- **Interpretation:**
  - 35.1 means typical error is 35.1 W/m²
  - If RMSE is much higher than MAE, you have occasional big mistakes
  - Lower is better

**R² (R-squared):**
- **Definition:** What fraction of variation is explained (0 to 1 scale)
- **Interpretation:**
  - 0.92 = explains 92% of the variation (excellent!)
  - 0.78 = explains 78% (good)
  - 0.45 = explains 45% (okay, room for improvement)
  - Generally: > 0.9 is excellent, 0.7–0.9 is good, < 0.7 needs work

### The Charts

#### **Actual vs Predicted Chart**
- **X-axis:** Time (from oldest data on left to newest on right)
- **Y-axis:** Solar irradiance in W/m² (0 to peak values)
- **Blue line:** Truth (what actually happened)
- **Orange line:** Prediction (what the model thought would happen)

**Good signs:**
- The lines are close together
- The orange line captures the peaks and valleys of the blue line
- Errors are about the same high and low

**Bad signs:**
- The lines diverge (get far apart)
- The model misses peaks or is consistently too high/low
- The orange line lags or leads the blue line

#### **Feature Importance Chart**
Shows 15 most important features for each horizon.

**What it tells you:**
- If `GHI_lag1` is at the top: "Recent history is the best predictor"
- If `Temperature_lag12` is high: "Temperature patterns 12 hours ago matter"
- If `month` is at the bottom: "Calendar month doesn't help much"

**Use case:** If a feature you think should be important isn't in the top 15, it might not be contributing—consider removing it later.

#### **Residual Analysis Chart**
Each dot represents one prediction:
- **X-axis:** What the model predicted
- **Y-axis:** How wrong the prediction was (error)
- **Red dashed line at y=0:** Represents perfect predictions

**Good patterns:**
- Random cloud of dots with no pattern
- Centered roughly on the red line (no systematic bias)
- Spread is consistent across all prediction levels

**Bad patterns:**
- U-shape or curve: Model does worse at certain prediction levels
- Dots systematically above/below the line: Model is biased
- Dots that get worse as predictions increase: Model loses confidence over higher values

---

## Troubleshooting

### Issue 1: "Missing required time column: Year"

**What went wrong:**
Your CSV file doesn't have a column exactly named "Year" (or Month, Day, Hour, Minute)

**How to fix:**
1. Open your CSV in Excel or a text editor
2. Check the first row (headers)
3. Make sure you have columns named exactly: `Year`, `Month`, `Day`, `Hour`, `Minute`
4. Save and try again

---

### Issue 2: "Neither GHI nor Clearsky GHI found"

**What went wrong:**
You don't have a column named "GHI" or "Clearsky GHI"

**How to fix:**
1. You need solar irradiance measurements
2. If your file has a column with a different name (like "Solar_Irradiance" or "irradiance"), rename it to `GHI`
3. Save and try again

---

### Issue 3: "Insufficient data: 0 rows after feature engineering"

**What went wrong:**
After creating features and removing incomplete rows, you have no data left

**Common causes:**
1. **Missing values:** Your columns have too many blanks or "N/A"
2. **Not enough rows:** You need at least 50-100 continuous rows
3. **All zeroes:** If GHI is all zeros (nighttime data), the model won't learn

**How to fix:**
1. **Fill missing values:** Use Excel to fill blanks with reasonable estimates
2. **More data:** Provide longer time periods (at least 1-2 weeks of continuous hourly data)
3. **Include daytime data:** Make sure you have sunny hours (9am-5pm) not just nighttime
4. **Check data quality:** Open your CSV and manually scan for obvious problems

---

### Issue 4: "Skipping horizon tplus168: insufficient valid samples"

**What went wrong:**
For the 7-day forecast, there's not enough complete data

**Why it happens:**
The 7-day forecast needs data from 7 days in the future to create targets. If your data is short, you lose the last 7 days.

**How to fix:**
- Provide more data (at least 1 month of continuous hourly data)

---

### Issue 5: Low R² scores (like 0.25)

**What it means:**
The model isn't explaining the variation well

**Common causes:**
1. **Not enough good data:** Weather station error, missing columns
2. **Trying to predict too far ahead:** 7-day solar forecasts are hard
3. **Missing important columns:** If you're missing Temperature, Humidity, etc., predictions suffer
4. **Data quality issues:** Sensors broken during certain periods

**How to fix:**
1. Add more weather variables (Pressure, Wind Speed, Dew Point, etc.)
2. For long-term (7-day) forecasts, accept lower accuracy—it's inherently hard
3. Check that your data is clean and continuous
4. Use only sunny periods if trying to improve accuracy (exclude nighttime)

---

### Issue 6: Downloaded CSV looks strange or incomplete

**What might be wrong:**
1. File didn't save properly
2. Browser issue

**How to fix:**
1. Try downloading again
2. Use a different browser
3. Check your Downloads folder

---

## Glossary of Terms

### Data Terms

**CSV:** Comma-Separated Values file. A simple text format for storing spreadsheet data. You can open it in Excel.

**Column:** A vertical list of data (like a spreadsheet column). Example: "Temperature" column contains all temperature values.

**Row:** A horizontal line of data (like a spreadsheet row). Example: One row = one hour of measurements.

**Feature:** A column of data used to make predictions. Example: "Temperature" is a feature.

**Target:** What you're trying to predict. In this program: future GHI values.

**Missing value / NaN:** A blank spot in data where no measurement was taken.

---

### Solar/Weather Terms

**GHI (Global Horizontal Irradiance):** Total solar energy hitting a horizontal surface (measured in W/m², watts per square meter).

**DHI (Diffuse Horizontal Irradiance):** Solar energy coming from the sky (not direct from the sun).

**DNI (Direct Normal Irradiance):** Solar energy coming directly from the sun's disk.

**Clearsky GHI:** Theoretical maximum GHI on a completely clear, cloudless day.

**Solar Zenith Angle:** The angle between the sun and the vertical (straight up). 0° = sun straight overhead (noon), 90° = sun at horizon (sunrise/sunset).

**Albedo:** How much light the ground reflects back (0 = black, 1 = white). Affects how much light bounces back to a solar panel.

---

### Machine Learning Terms

**Machine Learning Model:** A computer program that learns patterns from data to make predictions.

**Training:** The process of teaching the model by showing it examples.

**Validation/Testing:** Checking if the model learned correctly by testing it on new data.

**Feature Engineering:** Creating new, useful information from raw data.

**Lag:** A past value. "lag1" = value from 1 time period ago.

**Rolling Average/Mean:** Average calculated over a sliding window of recent values.

**Hyperparameters:** Tuning settings for the model (like "600 trees" or "learning rate 0.05").

---

### Performance Terms

**MAE (Mean Absolute Error):** Average prediction error in the same units as the target (W/m²).

**RMSE (Root Mean Squared Error):** Similar to MAE but emphasizes large errors more.

**R² (Coefficient of Determination):** How well the model explains variation in the data (0 = terrible, 1 = perfect).

**Bias:** Systematic error. Model consistently over-predicts or under-predicts.

**Variance/Volatility:** How much something fluctuates or changes.

**Residuals:** Prediction errors (actual value - predicted value).

---

### Technical Terms

**XGBoost:** A powerful machine learning algorithm used in this program. It learns by building 600 "decision trees" and combining them.

**Regression:** Predicting a continuous number (like "500 W/m²") instead of categories (like "sunny/cloudy").

**Split/Train-Test Split:** Dividing data into learning and testing portions.

**Epoch/Iteration:** One complete pass through training data.

**Overfitting:** When a model memorizes training data instead of learning general patterns (bad—it fails on new data).

---

## Summary

This program is a complete machine learning pipeline for solar forecasting:

1. **You provide:** Historical solar and weather data
2. **It does:** Clean data → Create features → Train models → Test accuracy → Show results
3. **You get:** Performance metrics, charts, and predictions you can use

The key insight: **Solar energy is mostly predictable in the short term (3-24h) because weather patterns change slowly, but long-term (7d) forecasts are harder because weather is chaotic.**

---

## Next Steps

**To use this effectively:**

1. **Gather good data:** At least 1 month of continuous hourly measurements
2. **Run the app:** Upload and see baseline performance
3. **Improve if needed:** Add more weather variables, clean data, tune parameters
4. **Deploy:** Use predictions in your operations or energy management system

**Questions?** Check the README.md for technical details or consult the inline code comments.

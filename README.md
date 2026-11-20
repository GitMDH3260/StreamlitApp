# Benford's Law and Two-Period KS Dashboard

A Streamlit app that lets you upload a CSV or Excel file, filter it, run a Benford analysis (first digit and first two
digits), test the results with the KS statistic (numpy), compare two time periods, and download everything.

## Run it on your computer
```
pip install -r requirements.txt
streamlit run app.py
```
(Per the Streamlit Intro notebook: save the code as `app.py`, then run `streamlit run app.py` in the terminal.)
Keep the `.streamlit` folder next to `app.py`. Its `config.toml` raises Streamlit's 200 MB upload limit to 1,000 MB, which
the 190 MB and 353 MB April checkbook files need.

## How to use it
1. **Upload.** Period A (required) and, to compare two time periods, period B (optional, for example the same month last
   year). CSV, .xlsx and .xlsm files work; Excel files use the first sheet. Only the first 1,048,575 rows of each file are read.
2. **Choose columns.** The amount to analyze, an optional date column, and two columns to filter by.
3. **Filters** (press *Apply filters and run analysis*):
   - Amount range. The default minimum of 0 keeps zero-dollar payments and drops negative payments.
   - Date range, one per period (needs dates written as YYYY-MM-DD).
   - Category filter 1 and category filter 2 (any column with 200 or fewer different values, for example department and payment method).
4. **Read the results** in the tabs: summary statistics (`.describe`), Benford first digit, Benford first two digits,
   and the two-period KS comparison.
5. **Download & share.** An Excel report (filters, statistics, tables, tests and charts), CSV tables and PNG charts.

## The statistics
- Digits come from `abs(amount) x 10,000,000,000`, as in the Data Viz and Two Groups notebook. A $0 payment has no
  leading digit, so it appears in the summary statistics but not in the Benford tables.
- **KS against Benford:** largest gap between the cumulative actual and Benford distributions; critical value 1.36 / sqrt(N)
  (N = number of first digits), as in the notebook.
- **KS between two periods:** the same numpy code with period B's distribution in place of Benford's. The decision uses the
  two-sample critical value 1.36 x sqrt((nA + nB) / (nA x nB)); the notebook's 1.36 / sqrt(N) (N = period A) is shown beside it.
- As in the notebook's two-digit code, amounts of 1,000,000 or more are written in scientific notation by the transformation
  and drop out of the first-two-digits table. The Summary tab reports how many.
- With hundreds of thousands of payments the critical value is tiny, so look at the size of the KS statistic as well.

## Share the dashboard (Streamlit Intro notebook, steps 3-5)
1. Put `app.py`, `requirements.txt` and `.streamlit/config.toml` in a GitHub repository.
2. Deploy the repository on Streamlit Community Cloud and share the app URL.
Streamlit Community Cloud has limited memory (about 1 GB), so for files as large as the April checkbook files run the app
locally, or deploy it with smaller files.

## Memory
Both files stay in memory while the app is open (about 1 GB for two 1,048,575-row files plus the uploaded bytes).

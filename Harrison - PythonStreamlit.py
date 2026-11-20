# =====================================================================================================
# Benford's Law & Two-Period KS Dashboard  (run with:  streamlit run app.py)
#
# PART 1 - analysis functions. These come from the course notebooks:
#   * calculate_benford_statistics, the first-two-digits block, the cleaned_amt transformation and the
#     "KS Statistic Using Numpy" cells are the Data Viz and Two Groups notebook's code.
#   * Filtering (boolean filters, .isin), .describe, pd.concat, .merge, .fillna, .nunique, read_csv /
#     read_excel and .plot(kind=...) with ax.set_title / set_xlabel / set_ylabel are Pandas notebook code.
# PART 2 - the Streamlit interface. The Streamlit Intro notebook teaches `import streamlit as st`,
#   st.write and `streamlit run app.py`; the other Streamlit widgets used below (uploader, sidebar, form,
#   tabs, download buttons, ...) come from the Streamlit documentation linked in that notebook.
# =====================================================================================================
import io
import datetime
import pandas as pd
import numpy as np # install numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st # install streamlit

ROW_LIMIT = 1048575          # only the first 1,048,575 rows of each file are used (Excel's row limit)
KS_NOTE = ('KS test at alpha = 0.05. A statistic above the critical value means the null hypothesis is rejected. '
           'With hundreds of thousands of payments the critical value is very small, so also look at the size of the KS statistic, '
           'not only whether it is rejected.')


# =====================================================================================================
# PART 1 - ANALYSIS (notebook code)
# =====================================================================================================
def calculate_benford_statistics(df, column, digits=1):
    # Data Viz and Two Groups notebook, unchanged
    counted1st = df[column].astype(str).str[0:digits].value_counts().to_frame().reset_index()
    counted1st.columns = ['digit', 'count']
    counted1st.sort_values(by='digit', inplace=True)
    counted1st['digit'] = pd.to_numeric(counted1st['digit'], errors='coerce')
    counted1st = counted1st[counted1st['digit'].notnull()]
    counted1st = counted1st[counted1st['digit'] != 0]
    counted1st['percent'] = counted1st['count'] / counted1st['count'].sum()
    counted1st['benford'] = counted1st['digit'].apply(lambda x: np.log10(1 + 1/int(x)))
    counted1st['diff'] = counted1st['percent'] - counted1st['benford']
    return counted1st


def calculate_two_digit_statistics(df, column, counted1st):
    # the notebook's "calculating first two digits" block, wrapped in a function
    counted1st2 = df[column].astype(str).str[0:2].value_counts().to_frame().reset_index()
    counted1st2.columns = ['digit', 'count']
    counted1st2 = counted1st2.sort_values(by='digit')
    counted1st2['digit'] = pd.to_numeric(counted1st2['digit'], errors='coerce')
    counted1st2 = counted1st2[counted1st2['digit'].notnull()]
    counted1st2 = counted1st2[counted1st2['digit'] > 9]
    counted1st2['percent'] = counted1st2['count'] / counted1st['count'].sum()
    counted1st2['benford'] = counted1st2['digit'].apply(lambda x: np.log10(1 + 1/int(x)))
    counted1st2['difference'] = counted1st2['percent'] - counted1st2['benford']
    return counted1st2


def complete_table(table, start, end, total):
    # After filtering, a digit can be missing. This gives every digit from start to end a row (count 0 if
    # absent) so that two periods line up when their cumulative distributions are compared.
    base = pd.DataFrame({'digit': list(range(start, end + 1))})
    found = pd.DataFrame({'digit': table['digit'].astype(int).values, 'count': table['count'].values})
    out = base.merge(found, on='digit', how='left')
    out = out.fillna(value=0)
    out['count'] = out['count'].astype(int)
    out['percent'] = out['count'] / total
    out['benford'] = out['digit'].apply(lambda x: np.log10(1 + 1/int(x)))
    out['difference'] = out['percent'] - out['benford']
    return out


def benford_for_period(df):
    # first-digit and first-two-digit tables for one period (None if there are no digits to count)
    if len(df) == 0:
        return None
    t1 = calculate_benford_statistics(df, 'CLEANED_AMT', digits=1)
    total_first = int(t1['count'].sum())
    if total_first == 0:
        return None
    t2 = calculate_two_digit_statistics(df, 'CLEANED_AMT', t1)
    full1 = complete_table(t1, 1, 9, total_first)
    full2 = complete_table(t2, 10, 99, total_first)
    return full1, full2, total_first


def ks_against_benford(table, n):
    # "KS Statistic Using Numpy" cells of the notebook (N = number of first digits)
    cum_diff = np.cumsum(table['percent'].values) - np.cumsum(table['benford'].values)
    max_diff = np.max(np.abs(cum_diff))
    critical_value = 1.36 / np.sqrt(n)
    return max_diff, critical_value


def ks_between_periods(table_a, table_b, n_a, n_b):
    # the same numpy code, with period B's distribution in place of the Benford column
    cum_diff = np.cumsum(table_a['percent'].values) - np.cumsum(table_b['percent'].values)
    max_diff = np.max(np.abs(cum_diff))
    critical_notebook = 1.36 / np.sqrt(n_a)                          # notebook formula, N = period A
    critical_two = 1.36 * np.sqrt((n_a + n_b) / (n_a * n_b))         # two-sample version
    return max_diff, critical_notebook, critical_two


# ----- data preparation and filtering (Pandas notebook techniques) -----
def read_uploaded(uploaded):
    if uploaded.name.lower().endswith('.csv'):
        df = pd.read_csv(uploaded, sep=',', nrows=ROW_LIMIT)
    else:
        df = pd.read_excel(uploaded, nrows=ROW_LIMIT)
    df.columns = df.columns.astype(str).str.upper()       # files from different years can differ in case
    return df


def prepare(df, amount_col, date_col):
    # helper columns, created before any filtering
    df['AMOUNT_NUM'] = pd.to_numeric(df[amount_col], errors='coerce')
    df['CLEANED_AMT'] = df['AMOUNT_NUM'].abs()*10000000000          # transformation from the notebook
    if date_col is not None:
        df['DATE_TEXT'] = df[date_col].astype(str).str[0:10]        # 'YYYY-MM-DD' text of the date


def date_bounds(df):
    # earliest and latest date as 'YYYY-MM-DD' text; None if the column does not hold such dates
    valid = df['DATE_TEXT'][df['DATE_TEXT'].str[4:5] == '-']
    valid = valid[valid.str[7:8] == '-']
    if len(valid) < 0.9 * len(df) or len(valid) == 0:
        return None
    return valid.min(), valid.max()


def category_candidates(df, skip):
    # columns with at most 200 different values can be used as a filter
    found = []
    for col in df.columns:
        if col in skip:
            continue
        n = df[col].nunique()
        if n > 0 and n <= 200:
            found.append(col)
    return found


def category_options(frames, col):
    options = []
    for df in frames:
        counts = df[col].astype(str).value_counts()
        for v in counts.index:
            if v not in options:
                options.append(v)
    return options


def apply_filters(df, amt_min, amt_max, date_range, cat_filters):
    # boolean filtering, one condition at a time (Pandas notebook); also records the rows left at each step
    steps = [('Rows read from the file', len(df))]
    out = df[df['AMOUNT_NUM'] >= amt_min]
    out = out[out['AMOUNT_NUM'] <= amt_max]
    steps.append(('Amount between ' + '{:,.2f}'.format(amt_min) + ' and ' + '{:,.2f}'.format(amt_max), len(out)))
    if date_range is not None:
        out = out[out['DATE_TEXT'] >= date_range[0]]
        out = out[out['DATE_TEXT'] <= date_range[1]]
        steps.append(('Payment date from ' + date_range[0] + ' to ' + date_range[1], len(out)))
    for col, selected, all_options in cat_filters:
        if len(selected) < len(all_options):
            out = out[out[col].astype(str).isin(selected)]
            steps.append((col + ' is one of ' + str(len(selected)) + ' selected values', len(out)))
    return out, steps


# ----- charts: pandas .plot(kind=...) as in the notebooks, saved as PNG bytes -----
def to_png(ax):
    plt.tight_layout()
    fig = ax.get_figure()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    return buf.getvalue()


def make_charts(labels, tables1, tables2):
    names = [x.replace('$', '') for x in labels]
    colors = ['#1F77B4', '#FF7F0E', '#2CA02C'] if len(names) == 2 else ['#1F77B4', '#2CA02C']
    charts = []

    # 1. first digit: actual vs Benford
    chart = pd.DataFrame({'digit': tables1[0]['digit'].values})
    for name, tbl in zip(names, tables1):
        chart[name] = tbl['percent'].values * 100
    chart['Benford'] = tables1[0]['benford'].values * 100
    ax = chart.plot(x='digit', kind='bar', figsize=(10, 5), color=colors)
    ax.set_title('Leading digit: actual vs. Benford')
    ax.set_xlabel('Leading digit')
    ax.set_ylabel('Share of payments (%)')
    charts.append(('first_digit_distribution', 'Leading digit: actual vs. Benford', to_png(ax)))

    # 2. first digit: actual minus Benford
    chart = pd.DataFrame({'digit': tables1[0]['digit'].values})
    for name, tbl in zip(names, tables1):
        chart[name + ' minus Benford'] = tbl['difference'].values * 100
    ax = chart.plot(x='digit', kind='bar', figsize=(10, 5), color=colors[0:len(names)])
    ax.set_title('Leading digit: difference between actual and Benford')
    ax.set_xlabel('Leading digit')
    ax.set_ylabel('Actual minus Benford (percentage points)')
    charts.append(('first_digit_difference', 'Leading digit: actual minus Benford', to_png(ax)))

    # 3. first two digits: actual vs Benford
    chart = pd.DataFrame({'digit': tables2[0]['digit'].values})
    for name, tbl in zip(names, tables2):
        chart[name] = tbl['percent'].values * 100
    chart['Benford'] = tables2[0]['benford'].values * 100
    ax = chart.plot(x='digit', kind='line', figsize=(11, 5), color=colors)
    ax.set_title('First two digits: actual vs. Benford')
    ax.set_xlabel('First two digits')
    ax.set_ylabel('Share of payments (%)')
    charts.append(('first_two_digits_distribution', 'First two digits: actual vs. Benford', to_png(ax)))

    # 4 and 5. cumulative distributions - the gap between the lines is what the KS statistic measures
    for key, title, tables, xlabel, marker in [
            ('first_digit_cumulative', 'Cumulative distribution of the leading digit', tables1, 'Leading digit', 'o'),
            ('first_two_digits_cumulative', 'Cumulative distribution of the first two digits', tables2, 'First two digits', None)]:
        chart = pd.DataFrame({'digit': tables[0]['digit'].values})
        for name, tbl in zip(names, tables):
            chart[name] = np.cumsum(tbl['percent'].values) * 100
        chart['Benford'] = np.cumsum(tables[0]['benford'].values) * 100
        ax = chart.plot(x='digit', kind='line', figsize=(10, 5), color=colors, marker=marker)
        ax.set_title(title + ' (KS statistic = largest gap between lines)')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Cumulative share of payments (%)')
        ax.set_ylim(0, 105)
        charts.append((key, title, to_png(ax)))
    return charts


def display_table(tbl, label):
    return pd.DataFrame({'Period': label,
                         'Digit': tbl['digit'].values,
                         'Count': tbl['count'].values,
                         'Actual (%)': tbl['percent'].values * 100,
                         'Benford (%)': tbl['benford'].values * 100,
                         'Difference (percentage points)': tbl['difference'].values * 100})


def build_results(frames, labels, funnels, loaded_rows):
    res = {'labels': labels}

    # descriptive statistics (.describe) of the amount for each period
    summary = pd.concat([f['AMOUNT_NUM'].describe() for f in frames], axis=1)
    summary.columns = labels
    res['summary'] = summary

    tables1, tables2, totals, overview = [], [], [], []
    for f, label, rows_read in zip(frames, labels, loaded_rows):
        result = benford_for_period(f)
        if result is None:
            res['error'] = label + ': no payments with a leading digit are left after the filters (zero-dollar payments have none).'
            return res
        tables1.append(result[0])
        tables2.append(result[1])
        totals.append(result[2])
        in_two = int(result[1]['count'].sum())
        overview.append({'Period': label,
                         'Rows read from the file': rows_read,
                         'Rows after filters': len(f),
                         'Zero-dollar payments': int((f['AMOUNT_NUM'] == 0).sum()),
                         'Negative payments': int((f['AMOUNT_NUM'] < 0).sum()),
                         'Payments with a leading digit (N)': result[2],
                         'Payments in the first-two-digits table': in_two,
                         'Left out of the first-two-digits table (amounts of 1,000,000 or more)': result[2] - in_two})
    res['overview'] = pd.DataFrame(overview)

    rows = []
    for label, steps in zip(labels, funnels):
        for step, n in steps:
            rows.append({'Period': label, 'Filter step': step, 'Rows remaining': n})
    res['filters'] = pd.DataFrame(rows)

    # statistical tests
    ks_rows = []
    for label, t1, t2, n in zip(labels, tables1, tables2, totals):
        for level, table in [('First digit', t1), ('First two digits', t2)]:
            max_diff, critical = ks_against_benford(table, n)
            decision = ('Different from Benford (reject the null hypothesis)' if max_diff > critical
                        else 'Consistent with Benford (cannot reject the null hypothesis)')
            ks_rows.append({'Test': 'Actual vs. Benford', 'Level': level, 'Period': label,
                            'N (period A)': n, 'N (period B)': None, 'KS statistic': max_diff,
                            'Critical value (notebook formula, 1.36/sqrt(N))': critical,
                            'Critical value (two-sample)': None, 'Decision': decision})
    if len(frames) == 2:
        for level, ta, tb in [('First digit', tables1[0], tables1[1]), ('First two digits', tables2[0], tables2[1])]:
            max_diff, crit_notebook, crit_two = ks_between_periods(ta, tb, totals[0], totals[1])
            decision = ('Statistically different (reject the null hypothesis)' if max_diff > crit_two
                        else 'Cannot reject the null hypothesis that the periods are the same')
            ks_rows.append({'Test': 'Period A vs. period B', 'Level': level, 'Period': labels[0] + ' vs. ' + labels[1],
                            'N (period A)': totals[0], 'N (period B)': totals[1], 'KS statistic': max_diff,
                            'Critical value (notebook formula, 1.36/sqrt(N))': crit_notebook,
                            'Critical value (two-sample)': crit_two, 'Decision': decision})
    res['ks'] = pd.DataFrame(ks_rows)

    res['first_digit'] = pd.concat([display_table(t, l) for t, l in zip(tables1, labels)])
    res['two_digits'] = pd.concat([display_table(t, l) for t, l in zip(tables2, labels)])
    res['charts'] = make_charts(labels, tables1, tables2)
    res['preview'] = [f.head(100) for f in frames]
    return res


def build_excel_report(res):
    # one shareable Excel workbook: filters, statistics, Benford tables, KS results and the charts
    from openpyxl.drawing.image import Image as XLImage
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        about = pd.DataFrame({'Item': ['Report', 'Created', 'Periods', 'Rows used', 'KS test'],
                              'Detail': ['Benford analysis and two-period KS comparison',
                                         datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                                         ' | '.join(res['labels']),
                                         'First ' + '{:,}'.format(ROW_LIMIT) + ' rows of each file, after the filters listed on the next sheet',
                                         KS_NOTE]})
        about.to_excel(writer, sheet_name='About', index=False)
        res['filters'].to_excel(writer, sheet_name='Filters used', index=False)
        res['overview'].to_excel(writer, sheet_name='Data overview', index=False)
        res['summary'].to_excel(writer, sheet_name='Summary statistics')
        res['first_digit'].to_excel(writer, sheet_name='First digit', index=False)
        res['two_digits'].to_excel(writer, sheet_name='First two digits', index=False)
        res['ks'].to_excel(writer, sheet_name='KS results', index=False)
        ws = writer.book.create_sheet('Charts')
        row = 1
        for key, title, png in res['charts']:
            ws.cell(row=row, column=1, value=title)
            img = XLImage(io.BytesIO(png))
            img.width, img.height = 750, 375
            ws.add_image(img, 'A' + str(row + 1))
            row += 22
        from openpyxl.worksheet.properties import PageSetupProperties
        ws.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)      # charts print on one page width
        ws.page_setup.fitToHeight = 0
        for sheet in writer.book.worksheets:
            sheet.column_dimensions['A'].width = 28
            for letter in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
                sheet.column_dimensions[letter].width = 24
    return buf.getvalue()


def csv_bytes(df, index=False):
    return df.to_csv(index=index).encode('utf-8')


# =====================================================================================================
# PART 2 - STREAMLIT INTERFACE
# =====================================================================================================
def load_slot(slot, uploaded):
    # keeps each uploaded file in memory so that moving a widget does not re-read a large file
    if uploaded is None:
        st.session_state.pop(slot, None)
        return None
    key = uploaded.name + '|' + str(uploaded.size)
    current = st.session_state.get(slot)
    if current is None or current['key'] != key:
        with st.spinner('Reading ' + uploaded.name + ' (first ' + '{:,}'.format(ROW_LIMIT) + ' rows) ...'):
            try:
                df = read_uploaded(uploaded)
            except Exception as error:
                st.sidebar.error('Could not read ' + uploaded.name + ': ' + str(error))
                return None
        st.session_state[slot] = {'key': key, 'name': uploaded.name, 'df': df, 'prepared': None}
        st.session_state.pop('results', None)
    return st.session_state[slot]


def guess_index(columns, keyword):
    for i, col in enumerate(columns):
        if keyword in col:
            return i
    return 0


def main():
    st.set_page_config(page_title='Benford & KS Dashboard', layout='wide')
    st.title("Benford's Law and Two-Period KS Dashboard")
    st.write('Upload a CSV or Excel file, filter it, then test whether the leading digits of the amounts follow '
             "Benford's Law and whether two time periods differ (KS statistic). Only the first "
             + '{:,}'.format(ROW_LIMIT) + ' rows of each file are used.')

    # ------------------------------ sidebar: 1. upload ------------------------------
    st.sidebar.header('1. Upload data')
    file_a = st.sidebar.file_uploader('Period A: CSV or Excel file (required)', type=['csv', 'xlsx', 'xlsm'], key='upload_a')
    file_b = st.sidebar.file_uploader('Period B: file to compare against (optional)', type=['csv', 'xlsx', 'xlsm'], key='upload_b')
    slot_a = load_slot('A', file_a)
    slot_b = load_slot('B', file_b)
    if slot_a is None:
        st.info('Upload a CSV or Excel file in the sidebar to begin. Add a second file (for example the same month '
                'from the prior year) to compare two periods.')
        st.write('**Sharing the app:** see the *Download & share* tab after you run an analysis.')
        return

    slots = [slot_a]
    labels = [st.sidebar.text_input('Label for period A', value=slot_a['name'].rsplit('.', 1)[0], key='label_a')]
    if slot_b is not None:
        slots.append(slot_b)
        label_b = st.sidebar.text_input('Label for period B', value=slot_b['name'].rsplit('.', 1)[0], key='label_b')
        if label_b == labels[0]:
            label_b = label_b + ' (comparison)'
        labels.append(label_b)

    # ------------------------------ sidebar: 2. columns ------------------------------
    st.sidebar.header('2. Choose columns')
    columns = list(slot_a['df'].columns)
    amount_col = st.sidebar.selectbox('Amount column (numbers to analyze)', columns,
                                      index=guess_index(columns, 'AMOUNT'), key='amount_col')
    date_options = ['(none)'] + columns
    date_pick = st.sidebar.selectbox('Date column (for the date filter)', date_options,
                                     index=guess_index(date_options, 'DATE') if guess_index(date_options, 'DATE') > 0 else 0,
                                     key='date_col')
    date_col = None if date_pick == '(none)' else date_pick

    if slot_b is not None and (amount_col not in slot_b['df'].columns or (date_col is not None and date_col not in slot_b['df'].columns)):
        st.sidebar.error('Period B does not have the selected amount/date column, so only period A is analyzed.')
        slots = [slot_a]
        labels = labels[0:1]

    for slot in slots:
        signature = (amount_col, date_col)
        if slot['prepared'] != signature:
            with st.spinner('Preparing ' + slot['name'] + ' ...'):
                prepare(slot['df'], amount_col, date_col)
            slot['prepared'] = signature
            slot['categories'] = None
            st.session_state.pop('results', None)

    frames_all = [s['df'] for s in slots]
    valid_share = frames_all[0]['AMOUNT_NUM'].notnull().mean()
    if valid_share == 0:
        st.error('The column ' + amount_col + ' has no numbers in it. Choose a different amount column in the sidebar.')
        return
    if valid_share < 0.5:
        st.sidebar.warning('Fewer than half of the values in ' + amount_col + ' are numbers. Check the amount column.')

    skip = [amount_col, 'AMOUNT_NUM', 'CLEANED_AMT', 'DATE_TEXT']
    if date_col is not None:
        skip.append(date_col)
    if slot_a.get('categories') is None:
        with st.spinner('Finding filterable columns ...'):
            slot_a['categories'] = category_candidates(slot_a['df'], skip)
    cat_columns = slot_a['categories']
    if slot_b is not None and len(slots) == 2:
        cat_columns = [c for c in cat_columns if c in slot_b['df'].columns]

    st.sidebar.header('3. Filters')
    cat_pick1 = cat_pick2 = '(none)'
    if len(cat_columns) > 0:
        pick_options = ['(none)'] + cat_columns
        default1 = pick_options.index('DEPARTMENT_ID') if 'DEPARTMENT_ID' in pick_options else 1
        default2 = pick_options.index('PAYMENT_METHOD') if 'PAYMENT_METHOD' in pick_options else (2 if len(pick_options) > 2 else 0)
        cat_pick1 = st.sidebar.selectbox('Category filter 1: column', pick_options, index=default1, key='cat_col_1')
        cat_pick2 = st.sidebar.selectbox('Category filter 2: column', pick_options, index=default2, key='cat_col_2')

    # ------------------------------ sidebar: filter form ------------------------------
    max_amount = float(max([f['AMOUNT_NUM'].max() for f in frames_all]))
    date_info = []
    if date_col is not None:
        for slot, label in zip(slots, labels):
            date_info.append(date_bounds(slot['df']))
    use_dates = date_col is not None and len(date_info) > 0 and all(d is not None for d in date_info)
    if date_col is not None and not use_dates:
        st.sidebar.warning('The date filter needs dates written as YYYY-MM-DD in column ' + date_col + '; it is switched off.')

    with st.sidebar.form('filters'):
        st.write('Filter 1: amount range')
        amt_min = st.number_input('Minimum amount', value=0.0, step=1.0, format='%.2f',
                                  help='0 keeps zero-dollar payments and drops negative payments. Use 0.01 to drop zero-dollar payments as well.')
        amt_max = st.number_input('Maximum amount', value=max_amount, step=1.0, format='%.2f')

        date_ranges = []
        if use_dates:
            st.write('Filter 2: date range (one range per period)')
            for (low, high), label in zip(date_info, labels):
                picked = st.date_input('Dates in ' + label, value=(datetime.date.fromisoformat(low), datetime.date.fromisoformat(high)),
                                       min_value=datetime.date.fromisoformat(low), max_value=datetime.date.fromisoformat(high),
                                       key='dates_' + label)
                if len(picked) == 2:
                    date_ranges.append((str(picked[0]), str(picked[1])))
                else:
                    date_ranges.append((low, high))

        cat_filters_ui = []
        for number, pick in [(1, cat_pick1), (2, cat_pick2)]:
            if pick != '(none)':
                options = category_options(frames_all, pick)
                st.write('Filter ' + str(number + 2) + ': ' + pick)
                chosen = st.multiselect(pick + ' (all selected = no filter)', options, default=options,
                                        key='cat_values_' + str(number) + '_' + pick)
                cat_filters_ui.append((pick, chosen, options))
        submitted = st.form_submit_button('Apply filters and run analysis')

    # ------------------------------ run the analysis ------------------------------
    if submitted:
        with st.spinner('Filtering and analyzing ...'):
            frames, funnels = [], []
            for i, slot in enumerate(slots):
                date_range = date_ranges[i] if use_dates else None
                out, steps = apply_filters(slot['df'], amt_min, amt_max, date_range, cat_filters_ui)
                frames.append(out)
                funnels.append(steps)
            st.session_state['results'] = build_results(frames, labels, funnels, [len(s['df']) for s in slots])

    res = st.session_state.get('results')
    if res is None:
        st.info('Data loaded: ' + ' | '.join([s['name'] + ' (' + '{:,}'.format(len(s['df'])) + ' rows, ' + str(len(s['df'].columns) - 2 - (1 if date_col else 0)) + ' columns)' for s in slots])
                + '. Choose the columns and filters in the sidebar, then press **Apply filters and run analysis**.')
        return
    if 'error' in res:
        st.error(res['error'])
        return

    # ------------------------------ results ------------------------------
    tab_summary, tab_first, tab_two, tab_periods, tab_share = st.tabs(
        ['Summary statistics', 'Benford: first digit', 'Benford: first two digits', 'Compare periods (KS)', 'Download & share'])
    two_periods = len(res['labels']) == 2
    ks = res['ks']
    charts = {key: png for key, title, png in res['charts']}
    num_format = st.column_config.NumberColumn(format='%.4f')
    table_config = {'Actual (%)': num_format, 'Benford (%)': num_format, 'Difference (percentage points)': num_format}

    def show_ks(test, level):
        rows = ks[(ks['Test'] == test) & (ks['Level'] == level)]
        for i in range(len(rows)):
            row = rows.iloc[i]
            st.write('**' + row['Period'] + '**')
            c1, c2, c3 = st.columns(3)
            c1.metric('KS statistic (maximum difference)', '{:.6f}'.format(row['KS statistic']))
            if test == 'Actual vs. Benford':
                c2.metric('Critical value 1.36/sqrt(N)', '{:.6f}'.format(row['Critical value (notebook formula, 1.36/sqrt(N))']))
                c3.metric('N (first digits)', '{:,}'.format(int(row['N (period A)'])))
            else:
                c2.metric('Critical value (two-sample)', '{:.6f}'.format(row['Critical value (two-sample)']))
                c3.metric('Critical value (notebook formula)', '{:.6f}'.format(row['Critical value (notebook formula, 1.36/sqrt(N))']))
            if row['KS statistic'] > (row['Critical value (two-sample)'] if test != 'Actual vs. Benford' else row['Critical value (notebook formula, 1.36/sqrt(N))']):
                st.warning(row['Decision'])
            else:
                st.success(row['Decision'])

    with tab_summary:
        st.subheader('Data overview')
        st.dataframe(res['overview'], hide_index=True)
        st.write('Rows remaining after each filter:')
        st.dataframe(res['filters'], hide_index=True)
        st.subheader('Descriptive statistics of the amount (.describe)')
        st.dataframe(res['summary'])
        st.subheader('Filtered data (first 100 rows)')
        for label, preview in zip(res['labels'], res['preview']):
            st.write('**' + label + '**')
            st.dataframe(preview.drop(columns=['AMOUNT_NUM', 'CLEANED_AMT']))

    with tab_first:
        st.subheader('First digit: actual vs. Benford')
        st.image(charts['first_digit_distribution'])
        st.image(charts['first_digit_difference'])
        st.dataframe(res['first_digit'], hide_index=True, column_config=table_config)
        st.subheader('KS test against Benford (first digit)')
        st.write(KS_NOTE)
        show_ks('Actual vs. Benford', 'First digit')

    with tab_two:
        st.subheader('First two digits: actual vs. Benford')
        st.image(charts['first_two_digits_distribution'])
        st.dataframe(res['two_digits'], hide_index=True, column_config=table_config)
        st.subheader('KS test against Benford (first two digits)')
        st.write(KS_NOTE)
        show_ks('Actual vs. Benford', 'First two digits')

    with tab_periods:
        if not two_periods:
            st.info('Upload a file for period B in the sidebar to test whether two periods are statistically different.')
        else:
            st.subheader('Are the two periods statistically different?')
            st.write('Period A (' + res['labels'][0] + ') is compared with period B (' + res['labels'][1] + '). The KS statistic is the largest '
                     'gap between the two cumulative distributions, as in the notebook. The decision uses the two-sample critical value; '
                     "the notebook's one-sample formula is shown for reference.")
            st.write('**First digit**')
            show_ks('Period A vs. period B', 'First digit')
            st.image(charts['first_digit_cumulative'])
            st.write('**First two digits**')
            show_ks('Period A vs. period B', 'First two digits')
            st.image(charts['first_two_digits_cumulative'])

    with tab_share:
        st.subheader('Download the results')
        st.write('Everything below reflects the filters you applied.')
        if 'excel' not in res:
            res['excel'] = build_excel_report(res)
        st.download_button('Download full report (Excel: statistics, tables, tests and charts)', data=res['excel'],
                           file_name='benford_ks_report.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        c1, c2 = st.columns(2)
        with c1:
            st.write('**Tables (CSV)**')
            st.download_button('Summary statistics (.describe)', data=csv_bytes(res['summary'], index=True), file_name='summary_statistics.csv', mime='text/csv')
            st.download_button('First-digit Benford table', data=csv_bytes(res['first_digit']), file_name='benford_first_digit.csv', mime='text/csv')
            st.download_button('First-two-digits Benford table', data=csv_bytes(res['two_digits']), file_name='benford_first_two_digits.csv', mime='text/csv')
            st.download_button('KS test results', data=csv_bytes(res['ks']), file_name='ks_results.csv', mime='text/csv')
            st.download_button('Filters used and data overview', data=csv_bytes(pd.concat([res['filters'], res['overview']])), file_name='filters_and_overview.csv', mime='text/csv')
        with c2:
            st.write('**Charts (PNG)**')
            for key, title, png in res['charts']:
                st.download_button(title, data=png, file_name=key + '.png', mime='image/png', key='png_' + key)
        st.subheader('Share the dashboard')
        st.write('Share the Excel report or the PNG/CSV files above to share these results. To share the dashboard itself, follow '
                 'the workflow in the Streamlit Intro notebook: save this file as app.py, push app.py and requirements.txt to a GitHub '
                 'repository, then deploy it on Streamlit Community Cloud and send the app URL. Large files (over about 200 MB) are '
                 'better handled by running the app locally; see the README.')


if __name__ == '__main__':
    main()

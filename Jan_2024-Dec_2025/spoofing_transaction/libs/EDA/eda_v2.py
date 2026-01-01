import warnings
import numpy as np
import pandas as pd
import plotly.express as px
from IPython.display import display
from scipy.stats import beta, ks_2samp
from sklearn.metrics import precision_recall_curve, confusion_matrix
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

# ======================================= Some first sample data ==============================================
"""
    | TransID  |    TransNB    |     TransDate       | Total_amount |        DVRStart     | Channel |     start_shift     |       end_shift     | EmployeeID | Register_Name | Loyalty_card | SiteName | SiteEmployee | exception_amount | T_OperatorID | T_PACID |      City    |      RegionName      | Division |                    exception_type              |   Payment   | CardNo |                                                            Desc                                                                         | link | Label | has_redeem_later | pre_card_id | pre_card_date | pre_card_amount  | pre_card_et | pre_card_item | next_card_id |    next_card_date   | next_card_amount  | next_card_et | next_card_item | pre_cashier_id  | pre_cashier_date  | pre_cashier_amount  | pre_cashier_et  | pre_cashier_item  | pre_cashier_payment | next_cashier_id | next_cashier_date  | next_cashier_amount | next_cashier_payment |
    |----------|---------------|---------------------|--------------|---------------------|---------|---------------------|---------------------|------------|---------------|--------------|----------|--------------|------------------|--------------|---------|--------------|----------------------|----------|------------------------------------------------|-------------|--------|-----------------------------------------------------------------------------------------------------------------------------------------|------|-------|------------------|-------------|---------------|------------------|-------------|---------------|--------------|---------------------|-------------------|--------------|----------------|-----------------|-------------------|---------------------|-----------------|-------------------|---------------------|-----------------|--------------------|---------------------|----------------------|
    | 19484930 | 1069520138330 | 2023-08-01 15:48:59 |   -142.33    | 2023-08-01 15:48:40 |   7.0   | 2023-08-01 15:45:33 | 2023-08-01 20:49:11 |   2365660  |      201      |              |    PRR   | PRR002365660 |      129.98      |      10382   |   46    | Porter Ranch | CA - Shannon Ansell  |    CA    |    return at RC, Cash Refund > $25             |    CASH     |        | Return: PROTEIN PLANT POWDER COMPLETE VANILLA \| -1 \| -62.9900; Return: PROTEIN POWDER ONE SHAKE US NATURAL LARGE OG \| -1 \| -66.9900 |      |   0   |                  |             |               |                  |             |               |  19484361.0  | 2023-08-01 15:50:52 |        14.22      |              |                |                 |                   |                     |                                                                                                                        |          CASH        |       
    | 19505630 | 1033327613897 | 2023-08-01 10:39:51 |     -8.9     | 2023-08-01 10:39:17 |   17.0  | 2023-08-01 07:13:26 | 2023-08-01 14:03:54 |    999890  |      276      |              |    VEN   | VEN000999890 |        8.9       |      10312   |   56    |    Venice    |  CA - Vincent Cruz   |    CA    |    W/O Customer, Return at SC                  | CreditDebit |        | Return: PARMESAN SARVECCHIO QUARTER \| -0.45 \| -8.9000                                                                                 |      |   0   |                  |             |               |                  |             |               |  19503050.0  | 2023-08-01 10:45:03 |         3.49      |  |  |  |  |  |                                                                                                                                                                                                     |          CASH        |
    | 19528046 | 1070127692157 | 2023-08-01 08:23:54 |    -25.35    | 2023-08-01 08:23:24 |   15.0  | 2023-08-01 07:45:50 | 2023-08-01 11:36:16 |    989651  |      276      |              |    LBR   | LBR000989651 |       22.99      |      10137   |   84    |  Long Beach  | CA - Steve Korhummel |    CA    |  W/O Customer, Return at SC, Cash Refund > $25 |    CASH     |        | Return: THERMOMETER GG CHEF PREC DGTL INST READ \| -1 \| -22.9900                                                                       |      |   0   |                  |             |               |                  |             |               |  19526447.0  | 2023-08-01 08:48:58 |        22.05      |  |  |  |  |  |                                                                                                                                                                                                     |        CreditDebit   |
"""

# ======================================= EDA & results ======================================================= #
# ========================= 1. Checking fraud-proportion in 2023 and later ==================================== #
def get_conclusion_fraud_rate(cur_df: pd.DataFrame):
    """
        -----------------------------------------------------
        Args:
            cur_df : current dataframe
        .................................................
        Result:
            Z-statistic: 76.22
            P-value: 0.0000e+00
            Conclusion: Reject H0 (Z = 76.22); fraud rate in 2023 is significantly different from post-2024, 
                        indicating a regime shift.        
    """
    # split data
    df_2023 = cur_df[cur_df['TransDate'].dt.year == 2023]
    df_post = cur_df[cur_df['TransDate'].dt.year >= 2024]

    # counts
    x = np.array([
        df_2023['Label'].sum(),
        df_post['Label'].sum()
    ])

    n = np.array([
        len(df_2023),
        len(df_post)
    ])

    z_stat, p_value = proportions_ztest(x, n, alternative='two-sided')

    print(f"Z-statistic: {z_stat:.2f}")
    print(f"P-value: {p_value:.4e}")
    alpha = 0.05
    z_crit = 1.96

    if abs(z_stat) > z_crit:
        print(
            f"Conclusion: Reject H0 (Z = {z_stat:.2f}); "
            "fraud rate in 2023 is significantly different from post-2024, "
            "indicating a regime shift."
        )
    else:
        print(
            f"Conclusion: Fail to reject H0 (Z = {z_stat:.2f}); "
            "no significant difference detected."
        )

    cur_df['Month-Year'] = cur_df['TransDate'].dt.strftime('%m-%Y')
    monthly_df = cur_df.groupby('Month-Year')['Label'].agg(txns='count', fraud_rate='mean', fraud_txns='sum').reset_index()
    monthly_df['Month-Year'] = pd.to_datetime(monthly_df['Month-Year'], format='%m-%Y')
    monthly_df = monthly_df.sort_values(by='Month-Year', ascending = True)
    
    fig = px.line( monthly_df,
            x = 'Month-Year', y = 'fraud_rate',
          
            markers = True, title = 'Fraud Rate Over Time'
    )

    fig.show()

    return p_value

# ================================== 2. Imbalance Analysis =================================================== #
def imbalance_analysis(cur_df_trunc):
    """
        Args:
            cur_df_trunc: truncated dataframe after 2023 because pval nearly to 0 in the previous function
        Result of this function:
                ================================================================================
                BAYESIAN SMOOTHING EXPLANATION
                ================================================================================
                Global fraud rate: 1.205% (dataset baseline)
                Smoothing strength (m): 50 = Pull small segments toward global rate
                Formula: (n*local_rate + 50*global_rate) / (n+50)
                Effect: n<50 → ~global_rate | n>50 → ~local_rate

                TOP 10 RISKY SEGMENTS (Smoothed Fraud Rate)
                (Ranked by smoothed_rate - STABLE ranking for action)
                RegionName	EmployeeID	SiteName	count	fraud_rate	smoothed_rate
                4361	CW - Jessica Nelson	2388688	KRK	32	1.000000	0.397589
                2143	CA - Steve Korhummel	2205806	MMM	42	0.833333	0.386981
                2544	CA - Vincent Cruz	903	WWD	28	1.000000	0.366696
                2547	CA - Vincent Cruz	906	WWD	27	1.000000	0.358471
                173	CA - David Stanley	2245509	SMT	25	1.000000	0.341364
                14998	SE - Tony Brunetto	903	PGA	24	1.000000	0.332463
                4974	CW - Joshua Crippen	2465021	BEL	31	0.806452	0.316078
                5787	CW - Nick Berger	1313849	TMP	21	1.000000	0.304257
                2277	CA - Steve Korhummel	2454800	PMD	20	1.000000	0.294318
                1550	CA - Shannon Ansell	2416914	BBK	19	1.000000	0.284091

                REGION FRAUD SUMMARY (Raw Rates)
                (All employees/sites in region - NO smoothing)
                RegionName	mean	count
                0	CA - David Stanley	0.017427	618
                1	CA - Robert Bohen	0.013200	568
                2	CA - Shannon Ansell	0.020989	843
                3	CA - Steve Korhummel	0.007906	512
                4	CA - Vincent Cruz	0.060171	629
                5	CW - Aaron Schryer	0.003160	501
                6	CW - Bobby Templet	0.019512	514
                7	CW - Jessica Nelson	0.009400	539
                8	CW - Joshua Crippen	0.002784	552
                9	CW - Nick Berger	0.015114	612
                10	MW - Eric Ortiz	0.010450	780
                11	MW - Jason Cody	0.019764	592
                12	MW - Jill Ellison	0.028749	633
                13	MW - Racheal Whittaker	0.008630	538
                14	NE - Bobby Thompson	0.012416	543
                15	NE - Bradley Zaretsky	0.001245	715
                16	NE - Janine Marin	0.013895	543
                17	NE - Leslie Charles	0.029687	64
                18	NE - Leslie Lorquet	0.010443	585
                19	NE - Randy Hall	0.020939	728
                20	NI - Anthony Auciello	0.007448	661
                21	NI - Chris Holmes	0.000000	131
                22	NI - Terri Marconi	0.006654	542
                23	SE - Christine Cunningham	0.014999	660
                24	SE - John Hudson	0.010104	681
                25	SE - Scott Mcentyre	0.018293	710
                26	SE - Tony Brunetto	0.015367	661
                27	WFM Offices and DCs	0.000000	10

                BUSINESS INSIGHTS:
                • Top-10 risky segments: 269 txns (33.8% avg rate)
                • 10 segments >3x global rate
                • ACTION: Investigate EmployeeIDs: [2388688, 2205806, 903, 906, 2245509, 903, 2465021, '1313849', 2454800, 2416914]        

    """
    global_rate = cur_df_trunc['Label'].mean()
    m = 50  # Bayesian smoothing parameter

    agg = (cur_df_trunc.groupby(['RegionName', 'EmployeeID', 'SiteName'])['Label']
        .agg(['count', 'mean']).reset_index()
        .rename(columns={'mean': 'fraud_rate'}))

    agg['smoothed_rate'] = (
        (agg['count'] * agg['fraud_rate'] + m * global_rate) /
        (agg['count'] + m)
    )

    top_risk = agg.nlargest(10, 'smoothed_rate')
    region_stats = agg.groupby('RegionName')['fraud_rate'].agg(['mean', 'count']).reset_index()

    # 🔥 EXPLANATION PRINTS
    print("="*80)
    print(" BAYESIAN SMOOTHING EXPLANATION")
    print("="*80)
    print(f" Global fraud rate: {global_rate:.3%} (dataset baseline)")
    print(f"  Smoothing strength (m): {m} = Pull small segments toward global rate")
    print(f" Formula: (n*local_rate + {m}*global_rate) / (n+{m})")
    print(f" Effect: n<50 → ~global_rate | n>50 → ~local_rate\n")

    print(" TOP 10 RISKY SEGMENTS (Smoothed Fraud Rate)")
    print("   (Ranked by smoothed_rate - STABLE ranking for action)")
    display(top_risk[['RegionName', 'EmployeeID', 'SiteName', 'count', 'fraud_rate', 'smoothed_rate']])

    print("\n REGION FRAUD SUMMARY (Raw Rates)")
    print("   (All employees/sites in region - NO smoothing)")
    display(region_stats)

    print("\n BUSINESS INSIGHTS:")
    print(f"• Top-10 risky segments: {top_risk['count'].sum():,} txns ({top_risk['smoothed_rate'].mean():.1%} avg rate)")
    print(f"• {len(top_risk[top_risk['smoothed_rate']>global_rate*3])} segments >3x global rate")
    print(f"• ACTION: Investigate EmployeeIDs: {top_risk['EmployeeID'].tolist()}")

# =============================== 3. Exception decomposition ================================================= #
def parse_exception_flags(df):
    """Binary flags from exception_type (Return at SC, W/O Customer, Cash Refund >$25)"""
    df['return_sc'] = df['exception_type'].str.contains('SC|Return at', na=False)
    df['no_customer'] = df['exception_type'].str.contains('W/O Customer', na=False)
    df['cash_refund_25'] = df['exception_type'].str.contains('Cash Refund > \$25', na=False)
    return df[['return_sc', 'no_customer', 'cash_refund_25', 'Label']].groupby(level=0).mean()

def summarize_binary_flag(df, flag_col, label_col='Label'):
    """
        Summarize key fraud statistics for a binary flag
    """
    df = parse_exception_flags(df)
    total_txn = len(df)
    total_fraud = df[label_col].sum()

    # coverage
    coverage = df[flag_col].mean()

    # fraud rate when flag = 1
    fraud_rate_flag = df.loc[df[flag_col] == 1, label_col].mean()

    # fraud rate when flag = 0
    fraud_rate_no_flag = df.loc[df[flag_col] == 0, label_col].mean()

    # baseline fraud rate
    baseline = df[label_col].mean()

    # lift
    lift = fraud_rate_flag / baseline if baseline > 0 else np.nan

    # recall contribution
    recall = (
        df.loc[(df[flag_col] == 1) & (df[label_col] == 1)].shape[0]
        / total_fraud
        if total_fraud > 0 else np.nan
    )

    # odds ratio
    eps = 1e-9
    odds_ratio = (
        (fraud_rate_flag + eps) / (1 - fraud_rate_flag + eps)
    ) / (
        (fraud_rate_no_flag + eps) / (1 - fraud_rate_no_flag + eps)
    )

    return {
        'coverage': coverage,
        'fraud_rate_flag': fraud_rate_flag,
        'baseline_fraud_rate': baseline,
        'lift': lift,
        'recall': recall,
        'odds_ratio': odds_ratio
    }

def exception_analysis(cur_df_trunc, flag_cols = ['return_sc', 'no_customer', 'cash_refund_25']):
    """
        Args:
            cur_df_trunc : current truncated dataframe
            flag_cols : list of flagged columns contains exception
        
        Output:
            | flag            | coverage | fraud_rate_flag | baseline_fraud_rate | lift     | recall   | odds_ratio |
            |-----------------|----------|-----------------|---------------------|----------|----------|------------|
            | return_sc       | 0.738120 | 0.006974        | 0.012046            | 0.578991 | 0.427365 | 0.259624   |
            | no_customer     | 0.000000 | NaN             | 0.012046            | NaN      | 0.000000 | NaN        |
            | cash_refund_25  | 0.079924 | 0.013366        | 0.012046            | 1.109580 | 0.088682 | 1.121873   |
        
        My conclusion:
            - `return_sc` có coverage rất cao (~74%) nhưng fraud rate (0.70%) thấp hơn đáng kể so với baseline (1.20%), với lift < 1 (0.58) 
                    và odds ratio ≪ 1 (0.26). Điều này cho thấy flag này hoạt động như tín hiệu anti-fraud / hành vi hợp lệ, không phù hợp 
                    làm rule phát hiện gian lận dương; nếu đưa vào model thì nên mang tác động âm.

            - `no_customer` có coverage bằng 0, dẫn đến fraud rate, lift và odds ratio không xác định. Điều này cho thấy flag không hoạt động 
                    (khả năng do lỗi dữ liệu hoặc logic gán flag) và cần được fix hoặc loại bỏ.

            - `cash_refund_25` có coverage thấp (~8%) và fraud rate (1.34%) chỉ nhỉnh hơn baseline (1.20%), với lift ≈ 1.11 và odds ratio ≈ 1.12. 
                    Flag này thể hiện giá trị dự báo độc lập rất yếu, ít hữu ích nếu dùng như rule và chỉ có vai trò hạn chế khi làm feature trong model.

            ==> Các exception flag nhìn chung mang lại giá trị phát hiện gian lận thấp: một flag là anti-fraud, một flag không hoạt động, và một flag chỉ có 
            lift rất nhỏ, cho thấy không có tín hiệu gian lận mạnh khi xét riêng lẻ.
    """
    summary = []

    for col in flag_cols:
        stats = summarize_binary_flag(cur_df_trunc, col)
        stats['flag'] = col
        summary.append(stats)

    summary_df = pd.DataFrame(summary).set_index('flag')
    
    return summary_df

# ============================== 4. Numeric Distributions (Fraud vs Non-Fraud) =============================== #
def compare_numeric_distributions(df, cols):
    """
        Args:
            df   : current truncated dataframe
            cols : list of colums
        
        Usages:
            compare_numeric_distributions(cur_df_trunc, 
                                         ['Total_amount', 'Channel', 'exception_amount'])
        Output:

            | feature            | ks_stat  | ks_pvalue        | fraud_q25 | fraud_q50 | fraud_q75 | nonfraud_q25 | nonfraud_q50 | nonfraud_q75 | median_ratio |
            |--------------------|----------|------------------|-----------|-----------|-----------|---------------|---------------|---------------|--------------|
            | Total_amount       | 0.291997 | 1.401087e-87     | -33.4900  | -6.080    | 2.6750    | -28.2900     | -13.39        | -4.87         | 0.454070     |
            | Channel            | 0.077541 | 2.209364e-06     | 8.0000    | 11.000    | 16.0000   | 7.0000       | 11.00         | 16.00         | 1.000000     |
            | exception_amount   | 0.277362 | 2.177354e-69     | 1.3575    | 14.625    | 42.8575   | 7.0225       | 15.99         | 30.00         | 0.914630     |

        Conclusion:
            - `Total_amount` shows strong distributional separation (KS ≈ 0.29, p ≪ 0.001), confirming it as a highly informative feature. 
                    Fraud transactions exhibit a distinct amount pattern compared to non-fraud, indicating this variable is suitable for 
                    both rule thresholding and model input, despite overlap in lower quantiles.

            - `Channel` displays minimal separation (KS ≈ 0.08) with nearly identical quantiles and median ratio ≈ 1, suggesting 
                    it carries little to no discriminatory power when treated as a numeric variable and should instead be handled as 
                    categorical or deprioritized.

            - `exception_amount` demonstrates meaningful separation (KS ≈ 0.28, p ≪ 0.001), with fraud transactions skewed toward higher
                    exception values, making it a useful risk indicator, though with some overlap around the median.

            => Total_amount and exception_amount provide strong discriminatory signals, while Channel offers negligible numeric separation and should not be modeled as a continuous feature.        
    """
    fraud = df[df["Label"] == 1]
    nonfraud = df[df["Label"] == 0]
    rows = []

    for col in cols:
        f = fraud[col].dropna()
        n = nonfraud[col].dropna()

        if len(f) == 0 or len(n) == 0:
            rows.append({
                "feature": col, "ks_stat": np.nan, "ks_pvalue": np.nan,
                "fraud_q25": np.nan, "fraud_q50": np.nan, "fraud_q75": np.nan,
                "nonfraud_q25": np.nan, "nonfraud_q50": np.nan, "nonfraud_q75": np.nan,
                "median_ratio": np.nan
            })
            continue

        ks_stat, p_val = ks_2samp(f, n)
        fq25, fq50, fq75 = f.quantile([0.25, 0.5, 0.75])
        nq25, nq50, nq75 = n.quantile([0.25, 0.5, 0.75])
        median_ratio = fq50 / nq50 if nq50 != 0 else np.inf

        rows.append({
            "feature": col, "ks_stat": ks_stat, "ks_pvalue": p_val,
            "fraud_q25": fq25, "fraud_q50": fq50, "fraud_q75": fq75,
            "nonfraud_q25": nq25, "nonfraud_q50": nq50, "nonfraud_q75": nq75,
            "median_ratio": median_ratio
        })

    return pd.DataFrame(rows).set_index("feature")

# ========================================= 5. Temporal Patterns ============================================= #
def analyze_time_features(df):
    """
        Extract + analyze hour/day patterns

        Args:
            df: current truncated dataframe
        
        Output:
            
            baseline: 0.012045618711403661        

            | hour  | 0   | 1   | 2   | 3   | 4   | 5   | 6     | 7     | 8     | 9     | 10    | 11    | 12    | 13    | 14    | 15    | 16    | 17    | 18    | 19    | 20    | 21    | 22    | 23    |
            |-------|-----|-----|-----|-----|-----|-----|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
            | Label | 0.0 | 0.0 | 0.0 | 0.4 | 0.0 | 0.0 | 0.053 | 0.015 | 0.022 | 0.016 | 0.012 | 0.012 | 0.011 | 0.010 | 0.009 | 0.010 | 0.012 | 0.012 | 0.010 | 0.013 | 0.013 | 0.009 | 0.011 | 0.222 |

        Conclusion:
            - The overall baseline fraud rate is ~1.2%, while most daytime hours (06:00 - 22:00) fluctuate narrowly around this level, 
              indicating no strong systematic intraday effect during normal operating hours.

            - Extreme spikes at 03:00 (40%) and 23:00 (22.2%) are observed; however, these occur in off-hours and are likely driven by 
              very low transaction volumes, suggesting statistical noise rather than stable fraud behavior.

            - Early morning hours (00:00 - 05:00) generally show near-zero activity, reinforcing the need to interpret rate spikes jointly with volume.

            - Mild elevations around 06:00 (5.3%) may warrant monitoring but are not sufficiently consistent to justify standalone time-based rules.

            => Hour-of-day effects appear weak and unstable, with isolated off-hour spikes likely attributable to low volume. Time features should therefore be used as secondary or supporting signals, preferably via binary indicators (e.g. late night / off-hours) rather than as continuous hour variables.
    """
    df['TransDate'] = pd.to_datetime(df['TransDate'])
    df['end_shift'] = pd.to_datetime(df['end_shift'])
    df['hour'] = df['TransDate'].dt.hour
    df['is_shift_end'] = (df['end_shift'] - df['TransDate']).dt.total_seconds() / 3600 < 1
    baseline = df['Label'].mean()
    
    res = df.groupby("hour")["Label"].mean().round(3).to_frame().T
    print("baseline:", baseline)
    with pd.option_context("display.max_columns", 24,
                           "display.width", None):
        display(res)

    return res

# ================================== 6. Employee/Site Risk Profiling ======================================== #
def get_confidence_interval(df, alpha=0.05, min_count=30):
    """
        df must have columns: count, sum, mean
    """
    df = df.copy()

    ci_low, ci_high = proportion_confint(
        count=df['sum'],
        nobs=df['count'],
        alpha=alpha,
        method='wilson'
    )

    df['ci_low'] = ci_low
    df['ci_high'] = ci_high
    df['ci_width'] = df['ci_high'] - df['ci_low']

    # mark unreliable rows
    df['reliable'] = df['count'] >= min_count

    return df

def profile_employee_site_risk(df, top_n=10):
    """Fraud rate + transaction volume by EmployeeID/SiteName"""
    emp_risk = df.groupby('EmployeeID')['Label'].agg(['count', 'sum', 'mean']).sort_values('mean', ascending=False)
    site_risk = df.groupby('SiteName')['Label'].agg(['count', 'sum', 'mean']).sort_values('mean', ascending=False)
    
    return emp_risk.head(top_n), site_risk.head(top_n)

def EmploySiteRisk_analysis(cur_df_trunc):
    """
        Result:
            >>> display(emp_risk_ci.sort_values('mean', ascending=False).head(10) )

                | EmployeeID | count | sum | mean | ci_low   | ci_high | ci_width | reliable |
                |------------|-------|-----|------|----------|---------|----------|----------|
                | 2545365    | 3     | 3   | 1.0  | 0.438503 | 1.0     | 0.561497 | False    |
                | 2356575    | 1     | 1   | 1.0  | 0.206549 | 1.0     | 0.793451 | False    |
                | 2337962    | 1     | 1   | 1.0  | 0.206549 | 1.0     | 0.793451 | False    |
                | 2418162    | 1     | 1   | 1.0  | 0.206549 | 1.0     | 0.793451 | False    |
                | 2473602    | 7     | 7   | 1.0  | 0.645670 | 1.0     | 0.354330 | False    |
                | 2488616    | 2     | 2   | 1.0  | 0.342380 | 1.0     | 0.657620 | False    |
                | 2497441    | 1     | 1   | 1.0  | 0.206549 | 1.0     | 0.793451 | False    |
                | 2495435    | 1     | 1   | 1.0  | 0.206549 | 1.0     | 0.793451 | False    |
                | 2491845    | 3     | 3   | 1.0  | 0.438503 | 1.0     | 0.561497 | False    |
                | 501755     | 1     | 1   | 1.0  | 0.206549 | 1.0     | 0.793451 | False    |

            >>> display(site_risk_ci.sort_values('mean', ascending=False).head(10) )

                | SiteName | count | sum | mean     | ci_low   | ci_high  | ci_width | reliable |
                |----------|-------|-----|----------|----------|----------|----------|----------|
                | WWD      | 486   | 149 | 0.306584 | 0.267242 | 0.348961 | 0.081719 | True     |
                | SHV      | 49    | 15  | 0.306122 | 0.195155 | 0.445279 | 0.250124 | True     |
                | TMP      | 170   | 36  | 0.211765 | 0.157068 | 0.279200 | 0.122133 | True     |
                | GWU      | 135   | 26  | 0.192593 | 0.134956 | 0.267240 | 0.132283 | True     |
                | MMM      | 199   | 38  | 0.190955 | 0.142401 | 0.251214 | 0.108813 | True     |
                | PMD      | 111   | 20  | 0.180180 | 0.119776 | 0.261981 | 0.142205 | True     |
                | KRK      | 197   | 34  | 0.172589 | 0.126215 | 0.231487 | 0.105272 | True     |
                | WOL      | 115   | 18  | 0.156522 | 0.101361 | 0.233888 | 0.132526 | True     |
                | BBK      | 409   | 61  | 0.149144 | 0.117891 | 0.186926 | 0.069035 | True     |
                | PGA      | 368   | 45  | 0.122283 | 0.092658 | 0.159711 | 0.067053 | True     |

        My conclusion:
            - `Employee-level results are not reliable`: The top employees all show 100% fraud rate, but with extremely low transaction counts (1-7 txns). 
                This indicates small-sample bias, making these signals statistically meaningless and unsuitable for risk ranking or action without 
                a minimum volume threshold.

            - `Site-level analysis is more informative`: Several sites (e.g. WWD, SHV, TMP) exhibit consistently elevated fraud rates (15-31%) 
                with substantial volumes, suggesting localized risk concentration that is operationally meaningful and warrants further investigation 
                or targeted controls.

            => Employee-level fraud rates are dominated by low-volume noise and should not be used without filtering on minimum activity, 
            whereas site-level fraud rates reveal meaningful and actionable risk patterns due to sufficient transaction volume.

        Quick recommendation (best practice):
            - Apply minimum count thresholds (e.g. count ≥ 30) before ranking employees or sites.

            - Use employee risk only as a supporting signal, never standalone.

            - Prioritize site-level monitoring, where both fraud rate and volume are material.
    """
    emp_risk, site_risk = profile_employee_site_risk(cur_df_trunc)

    emp_risk_ci = get_confidence_interval(emp_risk, min_count=30)
    site_risk_ci = get_confidence_interval(site_risk, min_count=30)

    display(emp_risk_ci.sort_values('mean', ascending=False).head(10) )
    display(site_risk_ci.sort_values('mean', ascending=False).head(10) )

# =================================== 7. Payment Method Analysis ============================================= #
def normalize_payment(payment):
    if pd.isna(payment):
        return "UNKNOWN"
    p = payment.upper()
    if "CASH" in p and "," not in p:
        return "CASH_ONLY"
    if "CASH" in p and "," in p:
        return "CASH_MIXED"
    if "GIFT" in p:
        return "GIFT_CARD"
    if "CREDIT" in p or "DEBIT" in p or "VISA" in p or "MASTER" in p or "AMEX" in p:
        return "CARD"
    return "OTHER"

def analyze_payment_risk(df, min_count = 50):
    """
        Result:
            >>> analyze_payment_risk(cur_df_trunc)

                    | payment_type | count | fraud | fraud_rate | ci_low   | ci_high  | ci_width | reliable |
                    |--------------|-------|-------|------------|----------|----------|----------|----------|
                    | CARD         | 47366 | 732   | 0.015454   | 0.014382 | 0.016605 | 0.002223 | True     |
                    | CASH_ONLY    | 19872 | 230   | 0.011574   | 0.010179 | 0.013158 | 0.002980 | True     |
                    | GIFT_CARD    | 7999  | 85    | 0.010626   | 0.008603 | 0.013120 | 0.004517 | True     |
                    | UNKNOWN      | 13886 | 125   | 0.009002   | 0.007561 | 0.010714 | 0.003153 | True     |
                    | CASH_MIXED   | 1089  | 5     | 0.004591   | 0.001963 | 0.010703 | 0.008740 | True     |
                    | OTHER        | 8081  | 7     | 0.000866   | 0.000420 | 0.001787 | 0.001367 | True     |
            
        Conclusion: 
            - CARD có fraud rate cao nhất (~1.55%), CI hẹp → risk thật, ổn định, vượt baseline → đáng giữ làm feature chính.

            - CASH_ONLY và GIFT_CARD có fraud rate thấp hơn CARD, CI vẫn hẹp → không phải driver chính, chỉ mức trung bình.

            - UNKNOWN thấp hơn baseline → ít giá trị cảnh báo, chủ yếu noise/thiếu info.

            - CASH_MIXED fraud rate thấp nhưng CI rất rộng do volume nhỏ → không kết luận được, cần theo dõi thêm.

            - OTHER fraud rate rất thấp, CI hẹp → near-zero risk, có thể bỏ.           
    """
    df = df.copy()
    df['payment_type'] = df['Payment'].apply(normalize_payment)

    agg = (
        df.groupby('payment_type')['Label']
          .agg(count='count', fraud='sum')
    )

    agg['fraud_rate'] = agg['fraud'] / agg['count']

    ci_low, ci_high = proportion_confint(
        agg['fraud'],
        agg['count'],
        method='wilson'
    )

    agg['ci_low'] = ci_low
    agg['ci_high'] = ci_high
    agg['ci_width'] = agg['ci_high'] - agg['ci_low']
    agg['reliable'] = agg['count'] >= min_count

    return agg.sort_values('fraud_rate', ascending=False)

# ==================================== 8. Channel Concentration ============================================== #
def summarize_channel_risk(df, min_count = 100, alpha = 0.05):
    """
        Args:
            df        : current truncated dataframe
            min_count : số giao dịch liên kết tối thiểu
            alpha     : significance level

        Result:
            🔴 HIGH-RISK CHANNELS (reliable)

                    | Channel | count | fraud_rate | lift     | ci_width |
                    |---------|-------|------------|----------|----------|
                    | 25.0    | 747   | 0.065596   | 5.445608 | 0.035695 |
                    | 30.0    | 740   | 0.037838   | 3.141212 | 0.027836 |
                    | 23.0    | 534   | 0.033708   | 2.798351 | 0.031224 |
                    | 9.0     | 5450  | 0.031376   | 2.604777 | 0.009277 |
                    | 49.0    | 370   | 0.024324   | 2.019350 | 0.032727 |
 
            ⚠️ UNRELIABLE CHANNELS (low volume)
                    | Channel | count | fraud_rate | lift     | ci_width |
                    |---------|-------|------------|----------|----------|
                    | 25.0    | 747   | 0.065596   | 5.445608 | 0.035695 |
                    | 30.0    | 740   | 0.037838   | 3.141212 | 0.027836 |
                    | 23.0    | 534   | 0.033708   | 2.798351 | 0.031224 |
                    | 9.0     | 5450  | 0.031376   | 2.604777 | 0.009277 |
                    | 49.0    | 370   | 0.024324   | 2.019350 | 0.032727 |

        Conclusion:

            - `Fraud risk` tập trung rõ rệt ở một nhóm nhỏ channel có volume đủ lớn. Các **channel 25, 30, 23, 9, 49** có lift `2-5× baseline`, 
                    CI tương đối hẹp → risk cấu trúc, đáng tin, nên giữ làm feature mạnh / rule candidate.

            - Những channel có fraud rate cực cao (ví dụ 46, 26, 28) đều volume rất thấp. CI rất rộng → pure noise, 
                    tuyệt đối không dùng để kết luận hay hành động.

            - Phần lớn channel còn lại nằm quanh hoặc dưới baseline. Lift ≈ 1 hoặc <1 → giá trị phân biệt thấp, 
                    chỉ hữu ích khi combine với feature khác.

            - Channel volume lớn nhưng fraud thấp (lift < 0.7). Có thể xem là protective signal hoặc dùng để down-weight risk.
    """

    baseline = df['Label'].mean()

    g = df.groupby('Channel')['Label'].agg(['count', 'sum'])
    g['fraud_rate'] = g['sum'] / g['count']
    g['lift'] = g['fraud_rate'] / baseline
    g['reliable'] = g['count'] >= min_count

    # Wilson CI
    ci_low, ci_high = proportion_confint(
        g['sum'], g['count'], alpha=alpha, method='wilson'
    )
    g['ci_low'] = ci_low
    g['ci_high'] = ci_high
    g['ci_width'] = g['ci_high'] - g['ci_low']

    high_risk = g[(g['reliable']) & (g['lift'] > 1.5)] \
        .sort_values('fraud_rate', ascending=False)

    noise = g[~g['reliable']]

    print("🔴 HIGH-RISK CHANNELS (reliable)")
    print(high_risk[['count','fraud_rate','lift','ci_width']].head(5))

    print("\n⚠️ UNRELIABLE CHANNELS (low volume)")
    display(noise.sort_values('fraud_rate', ascending=False).head(5))

    return g.sort_values('fraud_rate', ascending=False)

# ==================================== 9. Amount Link Analysis =============================================== #
def summarize_linked_card_risk_quantile(df, col, min_count=30, q=0.99):
    """
        Result:
            >>> pre_stats  = summarize_linked_card_risk_quantile(cur_df_trunc, 'pre_card_id', q=0.99)

                    HIGH-RISK LINKS for pre_card_id (top 1% fraud_rate) → No statistically meaningful high-risk links found.

            >>> next_stats = summarize_linked_card_risk_quantile(cur_df_trunc, 'next_card_id', q=0.99)

                🔗 HIGH-RISK LINKS for next_card_id (top 1% fraud_rate) → No statistically meaningful high-risk links found.

        Conclusion: Link-type feature (pre_card_id, next_card_id) có 3 đặc điểm:

            - Card ID: rất phân mảnh → mỗi card chỉ xuất hiện vài lần
            - Fraud hiếm: → để lọt vào top 1% fraud_rate + count ≥ 30 là cực khó
            - Fraud không “lan truyền” theo card trong dataset này → khác retail fraud kiểu stolen card spam liên tiếp

        => Không tồn tại cụm card có fraud_rate đủ cao + đủ volume để kết luận contagion.
        => Đây là insight âm (negative result) nhưng rất giá trị.
    """

    baseline = df['Label'].mean()

    g = df.groupby(col)['Label'].agg(['count','sum'])
    g['fraud_rate'] = g['sum'] / g['count']
    g['lift'] = g['fraud_rate'] / baseline
    g['reliable'] = g['count'] >= min_count

    # CI
    ci_low, ci_high = proportion_confint(
        g['sum'], g['count'], method='wilson'
    )
    g['ci_width'] = ci_high - ci_low

    # adaptive threshold
    fr_threshold = g.loc[g['reliable'], 'fraud_rate'].quantile(q)

    high_risk = g[
        (g['reliable']) &
        (g['fraud_rate'] >= fr_threshold) &
        (g['lift'] > 1)
    ].sort_values('fraud_rate', ascending=False)

    print(f"🔗 HIGH-RISK LINKS for {col} (top {int((1-q)*100)}% fraud_rate)")
    if high_risk.empty:
        print("→ No statistically meaningful high-risk links found.")
    else:
        print(high_risk[['count','fraud_rate','lift','ci_width']].head())

    return g.sort_values('fraud_rate', ascending=False)

# ===================================== 10. Operator/PAC Risk ================================================ #
def summarize_operator_pac_risk(df, min_count=50, q=0.99):
    """
        Result:
            🧑‍💼 HIGH-RISK OPERATOR-PAC COMBOS :

                    | T_OperatorID | T_PACID | count | fraud_rate | lift      | ci_width |
                    |--------------|---------|-------|------------|-----------|----------|
                    | 34080        | 590     | 66    | 0.181818   | 15.094134 | 0.184266 |
                    | 81844        | 469     | 55    | 0.181818   | 15.094134 | 0.201428 |
                    | 25662        | 1141    | 59    | 0.033898   | 2.814161  | 0.106090 |

        Conclusion:

            - Chỉ 2-3 Operator_PAC combos có risk thật (≈ 18% fraud, ~15x baseline, volume đủ) → audit ngay.
            - Hầu hết fraud = 100% là noise do volume rất nhỏ → bỏ.
            - Không có collusion diện rộng, risk chỉ tập trung cục bộ.

            => Action: flag vài combo risk cao; không dùng raw Operator_PAC cho model, chỉ dùng flag hoặc tách Operator/PAC riêng.
    """

    baseline = df['Label'].mean()

    g = df.groupby(['T_OperatorID','T_PACID'])['Label'].agg(['count','sum'])
    g['fraud_rate'] = g['sum'] / g['count']
    g['lift'] = g['fraud_rate'] / baseline
    g['reliable'] = g['count'] >= min_count

    ci_low, ci_high = proportion_confint(
        g['sum'], g['count'], method='wilson'
    )
    g['ci_width'] = ci_high - ci_low

    # adaptive threshold
    fr_q = g.loc[g['reliable'], 'fraud_rate'].quantile(q)

    high_risk = g[
        (g['reliable']) &
        (g['fraud_rate'] >= fr_q) &
        (g['lift'] > 1)
    ].sort_values('fraud_rate', ascending=False)

    print("🧑‍💼 HIGH-RISK OPERATOR–PAC COMBOS")
    if high_risk.empty:
        print("→ No statistically stable high-risk Operator–PAC patterns found.")
    else:
        print(high_risk[['count','fraud_rate','lift','ci_width']].head())

    return g.sort_values('fraud_rate', ascending=False)

# ======================================== 11. Rule-Based Feature Validation ================================= #
def validate_spoofing_rules(df):
    """
        Result:
            🧪 SPOOFING RULE VALIDATION :

                cash_gt25    | Recall= 9.9% | Precision= 1.5% | Lift=1.27x | Count=7627
                sc_return    | Recall=30.6% | Precision= 0.6% | Lift=0.49x | Count=60901
                no_cust      | Recall= 0.0% | Precision= 0.0% | Lift=0.00x | Count=0
                high_channel | Recall=28.0% | Precision= 1.2% | Lift=0.99x | Count=27671

        Conclusion:
            - cash_gt25 → weak positive signal : lift > 1 nhưng precision thấp → dùng làm feature, KHÔNG làm rule block
            - sc_return → anti-signal: bắt nhiều fraud nhưng fraud rate thấp hơn baseline → noisy
            - high_channel → neutral
            - no_cust → data rỗng → drop

            => Không rule nào đủ mạnh để standalone → đúng hướng chuyển sang Bayes.
    """
    baseline = df['Label'].mean()
    total_fraud = df['Label'].sum()
    df['Total_amount_abs'] = df['Total_amount'].apply(np.abs)

    rules = {
        'cash_gt25': (df['Total_amount_abs'] > 25) & (df['Payment'] == 'CASH'),
        'sc_return': df['exception_type'].str.contains('SC', na=False),
        'no_cust': df['exception_type'].str.contains('W/O Customer', na=False),
        'high_channel': df['Channel'] > 15
    }

    print("🧪 SPOOFING RULE VALIDATION\n")

    for name, rule in rules.items():
        hit = df[rule]
        fraud_hit = hit['Label'].sum()

        recall = fraud_hit / total_fraud if total_fraud > 0 else 0
        precision = fraud_hit / len(hit) if len(hit) > 0 else 0
        lift = precision / baseline if baseline > 0 else 0

        print(
            f"{name:12s} | "
            f"Recall={recall:5.1%} | "
            f"Precision={precision:5.1%} | "
            f"Lift={lift:4.2f}x | "
            f"Count={len(hit)}"
        )

# ========================================= 12. Bayesian survey ============================================== #
def bayes_category_survey(df, col, min_count=100):
    """
        Result:
            >>> bayes_category_survey(cur_df_trunc, 'Payment')

                    | Payment             | count | fraud | p_fraud_given_x  | lift     | reliable |
                    |---------------------|-------|-------|------------------|----------|----------|
                    | CreditDebit         | 7349  | 527   | 0.071710         | 5.953238 | True     |
                    | CASH                | 19863 | 230   | 0.011579         | 0.961289 | True     |
                    | Gift Card           | 7744  | 77    | 0.009943         | 0.825460 | True     |
                    | CASH, Coupon        | 106   | 1     | 0.009434         | 0.783186 | True     |
                    | CASH, Visa          | 117   | 1     | 0.008547         | 0.709553 | True     |
                    | MasterCard          | 7064  | 39    | 0.005521         | 0.458337 | True     |
                    | Debit               | 5167  | 26    | 0.005032         | 0.417740 | True     |
                    | Visa                | 27625 | 139   | 0.005032         | 0.417718 | True     |
                    | Discover            | 1132  | 4     | 0.003534         | 0.293349 | True     |
                    | CASH, CreditDebit   | 597   | 2     | 0.003350         | 0.278116 | True     |
                    | American Express    | 5185  | 3     | 0.000579         | 0.048033 | True     |
                    | Amz Crdt Crd        | 139   | 0     | 0.000000         | 0.000000 | True     |
                    | EBT Food Stamps     | 1374  | 0     | 0.000000         | 0.000000 | True     |
                    | Void transaction    | 123   | 0     | 0.000000         | 0.000000 | True     |

             => Conclusion: Chỉ có 1 signal mạnh, phần còn lại là anti-fraud

                - CreditDebit → fraud rate 7.17%, lift ~6x → 🔥 HIGH-RISK payment
                - CASH → lift ~1 → neutral (không xấu như trực giác)
                - Visa / Master / Debit / Amex → lift < 0.5 → anti-fraud

                👉 Decision :
                    * Giữ biến (cột) Payment
                    * Encode riêng CreditDebit = high-risk flag
                    * Các payment còn lại: negative / neutral weight

            >>> bayes_category_survey(cur_df_trunc, 'Channel')

                    | Channel | count | fraud | p_fraud_given_x | lift     | reliable |
                    |---------|-------|-------|-----------------|----------|----------|
                    | 25.0    | 747   | 49    |  0.065596       | 5.445608 | True     |
                    | 30.0    | 740   | 28    |  0.037838       | 3.141212 | True     |
                    | 23.0    | 534   | 18    |  0.033708       | 2.798351 | True     |
                    | 9.0     | 5450  | 171   |  0.031376       | 2.604777 | True     |
                    | 49.0    | 370   | 9     |  0.024324       | 2.019350 | True     |
                    | 16.0    | 4282  | 92    |  0.021485       | 1.783660 | True     |
                    | 10.0    | 4637  | 88    |  0.018978       | 1.575493 | True     |
                    | 40.0    | 424   | 7     |  0.016509       | 1.370576 | True     |
                    | 15.0    | 3943  | 63    |  0.015978       | 1.326431 | True     |
                    | 32.0    | 384   | 5     |  0.013021       | 1.080960 | True     |
                    | 4.0     | 4661  | 57    |  0.012229       | 1.015235 | True     |
                    | 12.0    | 5051  | 56    |  0.011087       | 0.920410 | True     |
                    | 14.0    | 3675  | 40    |  0.010884       | 0.903594 | True     |
                    | 24.0    | 835   | 9     |  0.010778       | 0.894802 | True     |
                    | 6.0     | 4225  | 45    |  0.010651       | 0.884213 | True     |
                    | 5.0     | 6571  | 69    |  0.010501       | 0.871743 | True     |
                    | 22.0    | 2775  | 29    |  0.010450       | 0.867573 | True     |
                    | 39.0    | 289   | 3     |  0.010381       | 0.861776 | True     |
                    | 7.0     | 4392  | 43    |  0.009791       | 0.812787 | True     |
                    | 13.0    | 6821  | 59    |  0.008650       | 0.718083 | True     |
                    | 8.0     | 6480  | 56    |  0.008642       | 0.717437 | True     |
                    | 51.0    | 138   | 1     |  0.007246       | 0.601578 | True     |
                    | 17.0    | 2600  | 18    |  0.006923       | 0.574738 | True     |
                    | 11.0    | 5965  | 40    |  0.006706       | 0.556699 | True     |
                    | 3.0     | 3195  | 20    |  0.006260       | 0.519673 | True     |
                    | 34.0    | 560   | 3     |  0.005357       | 0.444738 | True     |
                    | 2.0     | 1877  | 9     |  0.004795       | 0.398061 | True     |
                    | 33.0    | 231   | 1     |  0.004329       | 0.359384 | True     |
                    | 20.0    | 4584  | 19    |  0.004145       | 0.344096 | True     |
                    | 18.0    | 2097  | 7     |  0.003338       | 0.277122 | True     |
                    | 19.0    | 1880  | 5     |  0.002660       | 0.220792 | True     |
                    | 1.0     | 957   | 2     |  0.002090       | 0.173496 | True     |
                    | 21.0    | 1986  | 3     |  0.001511       | 0.125404 | True     |
                    | 36.0    | 161   | 0     |  0.000000       | 0.000000 | True     |
                    | 31.0    | 170   | 0     |  0.000000       | 0.000000 | True     |
                    | 27.0    | 557   | 0     |  0.000000       | 0.000000 | True     |
                    | 42.0    | 100   | 0     |  0.000000       | 0.000000 | True     |
                    | 47.0    | 188   | 0     |  0.000000       | 0.000000 | True     |
                    | 43.0    | 275   | 0     |  0.000000       | 0.000000 | True     |
                    | 48.0    | 297   | 0     |  0.000000       | 0.000000 | True     |
                    | 56.0    | 111   | 0     |  0.000000       | 0.000000 | True     |

             => Conclusion:
                - Fraud tập trung mạnh vào 1 nhóm nhỏ channel
                - High-risk rõ ràng: 25, 30, 23, 9
                    → fraud rate 3-6%
                    → lift 2.6x - 5.4x
                - Neutral zone:
                    Channel ~10-16
                    → lift ~1-1.7
                - Safe / anti-fraud:
                    Channel < 8, > 18 (đa số)
                    → lift < 0.7

                👉 Decision:
                    * Không threshold kiểu Channel > k
                    * Dùng Bayes / categorical encoding
                    * Có thể tạo: high_risk_channel = {25,30,23,9}

            >>> bayes_category_survey(cur_df_trunc, 'SiteName')

                    | SiteName | count | fraud | p_fraud_given_x | lift       | reliable |
                    |----------|-------|-------|-----------------|------------|----------|
                    | WWD      | 486   | 149   |  0.306584       | 25.451940  | True     |
                    | TMP      | 170   | 36    |  0.211765       | 17.580227  | True     |
                    | GWU      | 135   | 26    |  0.192593       | 15.988601  | True     |
                    | MMM      | 199   | 38    |  0.190955       | 15.852633  | True     |
                    | PMD      | 111   | 20    |  0.180180       | 14.958151  | True     |
                    | ...      | ...   | ...   |  ...            | ...        | ...      |
                    | LAJ      | 440   | 0     |  0.000000       | 0.000000   | True     |
                    | KWD      | 108   | 0     |  0.000000       | 0.000000   | True     |
                    | KTL      | 331   | 0     |  0.000000       | 0.000000   | True     |
                    | KNW      | 125   | 0     |  0.000000       | 0.000000   | True     |
                    | MRL      | 213   | 0     |  0.000000       | 0.000000   | True     |

                397 rows x 5 columns

             => Conclusion: Đây là strongest signal toàn dataset
                - Top sites:
                    * WWD → fraud 30.6%, lift 25x
                    * TMP / GWU / MMM / PMD → lift 15–18x

                - Đa số site còn lại: fraud = 0 → safe sites
                    👉 Interpret đúng (rất quan trọng)
                    ❌ Không phải “site xấu”
                    ✅ Là local process / staff / exploit pattern

            👉 Decision:
                * Giữ SiteName
                * BẮT BUỘC shrinkage / Bayesian smoothing
                * Không dùng raw fraud rate
    """
    baseline = df['Label'].mean()

    g = (
        df.groupby(col)['Label']
          .agg(['count','sum'])
          .rename(columns={'sum':'fraud'})
    )

    g['p_fraud_given_x'] = g['fraud'] / g['count']
    g['lift'] = g['p_fraud_given_x'] / baseline
    g['reliable'] = g['count'] >= min_count

    return (
        g[g['reliable']]
        .sort_values('lift', ascending=False)
    )

# ======================================= 13. Bayesian exploratory analysis ================================== #
class BayesianRiskProfiler:
    def __init__(
        self,
        target_col='Label',
        alpha=1.0,
        beta_=1.0,
        min_count=30,
        ci_level=0.95
    ):
        self.target_col = target_col
        self.alpha = alpha
        self.beta_ = beta_
        self.min_count = min_count
        self.ci_level = ci_level

        self.baseline_p = None
        self.baseline_logit = None
        self.tables = {}

    @staticmethod
    def _logit(p, eps=1e-6):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def fit_category(self, df, col):
        g = df.groupby(col)[self.target_col].agg(['count', 'sum'])
        g.rename(columns={'sum': 'fraud'}, inplace=True)

        g['p_fraud'] = (
            g['fraud'] + self.alpha
        ) / (
            g['count'] + self.alpha + self.beta_
        )

        g['logit'] = self._logit(g['p_fraud'])
        g['lift'] = g['p_fraud'] / self.baseline_p

        # Credible Interval
        a = g['fraud'] + self.alpha
        b = g['count'] - g['fraud'] + self.beta_
        lo, hi = beta.interval(self.ci_level, a, b)
        g['ci_low'] = lo
        g['ci_high'] = hi
        g['ci_width'] = hi - lo

        g['reliable'] = g['count'] >= self.min_count

        self.tables[col] = g.sort_values('lift', ascending=False)
        return self.tables[col]

    def fit(self, df, categorical_cols):
        self.baseline_p = df[self.target_col].mean()
        self.baseline_logit = self._logit(self.baseline_p)

        for col in categorical_cols:
            self.fit_category(df, col)

        return self

    def score(self, df, weights=None):
        if weights is None:
            weights = {col: 1.0 for col in self.tables.keys()}

        score = np.full(len(df), self.baseline_logit)

        for col, table in self.tables.items():
            w = weights.get(col, 1.0)

            delta = (
                df[col]
                .map(table['logit'])
                .fillna(self.baseline_logit)
                - self.baseline_logit
            )

            score += w * delta

        return self._sigmoid(score)
    
def get_eda_result(cur_df_trunc):
    """
        Results:
            =============================================================================================================
            # 1. EDA inspects:
            -------------------------------------------------------------------------------------------------------------
            >>> profiler.tables['SiteName'].head(10)

                | SiteName | count | fraud | p_fraud | logit     | lift      | ci_low   | ci_high  | ci_width | reliable |
                |----------|-------|-------|---------|-----------|-----------|----------|----------|----------|----------|
                | YNG      | 1     | 0     | 0.333333| -0.693147 | 27.672579 | 0.012579 | 0.841886 | 0.829307 | False    |
                | ARG      | 1     | 0     | 0.333333| -0.693147 | 27.672579 | 0.012579 | 0.841886 | 0.829307 | False    |
                | SHV      | 49    | 15    | 0.313725| -0.782759 | 26.044780 | 0.195204 | 0.446082 | 0.250878 | False    |
                | WWD      | 486   | 149   | 0.307377| -0.812411 | 25.517747 | 0.267252 | 0.348998 | 0.081746 | True     |
                | ISP      | 2     | 0     | 0.250000| -1.098612 | 20.754434 | 0.008404 | 0.707598 | 0.699194 | False    |
                | VIC      | 2     | 0     | 0.250000| -1.098612 | 20.754434 | 0.008404 | 0.707598 | 0.699194 | False    |
                | TMP      | 170   | 36    | 0.215116| -1.294357 | 17.858467 | 0.157157 | 0.279347 | 0.122190 | True     |
                | GWU      | 135   | 26    | 0.197080| -1.404643 | 16.361160 | 0.135096 | 0.267434 | 0.132337 | True     |
                | MMM      | 199   | 38    | 0.194030| -1.424035 | 16.107919 | 0.142492 | 0.251332 | 0.108840 | True     |
                | PMD      | 111   | 20    | 0.185841| -1.477266 | 15.428075 | 0.119974 | 0.262227 | 0.142253 | True     |

            >>> profiler.tables['Payment'].head(10)

                | Payment                                   | count | fraud | p_fraud | logit     | lift      | ci_low   | ci_high  | ci_width | reliable |
                |-------------------------------------------|-------|-------|---------|-----------|-----------|----------|----------|----------|----------|
                | Gift Card, Gift Card, Gift Card           | 2     | 2     | 0.750000| 1.098612  | 62.263302 | 0.292402 | 0.991596 | 0.699194 | False    |
                | Visa, Visa, Visa, Visa, Visa              | 1     | 1     | 0.666667| 0.693147  | 55.345158 | 0.158114 | 0.987421 | 0.829307 | False    |
                | American Express, MasterCard              | 1     | 0     | 0.333333| -0.693147 | 27.672579 | 0.012579 | 0.841886 | 0.829307 | False    |
                | CASH, EBT Food Stamps, Gift Card          | 1     | 0     | 0.333333| -0.693147 | 27.672579 | 0.012579 | 0.841886 | 0.829307 | False    |
                | CASH, Coupon, MasterCard                  | 1     | 0     | 0.333333| -0.693147 | 27.672579 | 0.012579 | 0.841886 | 0.829307 | False    |
                | CASH, Coupon, American Express            | 1     | 0     | 0.333333| -0.693147 | 27.672579 | 0.012579 | 0.841886 | 0.829307 | False    |
                | CASH, Coupon, Discover                    | 1     | 0     | 0.333333| -0.693147 | 27.672579 | 0.012579 | 0.841886 | 0.829307 | False    |
                | CASH, Amz Crdt Crd, Gift Card             | 1     | 0     | 0.333333| -0.693147 | 27.672579 | 0.012579 | 0.841886 | 0.829307 | False    |
                | CASH, EBT Cash, EBT Food Stamps           | 1     | 0     | 0.333333| -0.693147 | 27.672579 | 0.012579 | 0.841886 | 0.829307 | False    |
                | CASH, Visa, Debit                         | 1     | 0     | 0.333333| -0.693147 | 27.672579 | 0.012579 | 0.841886 | 0.829307 | False    |

            >>> Conclusion:
                    - Fraud mang tính cấu trúc theo Site, không phải noise
                    - Payment:
                            * Các combo 1-2 record → bỏ
                            * Payment “core” (CreditDebit, CASH, Gift Card) đã được xử lý tốt ở prior cấp cao hơn → đúng hướng.

            =============================================================================================================
            # 2. Metric evaluation.
            -------------------------------------------------------------------------------------------------------------
            >>> print(...)

                Median bayes_risk | Fraud=0.2980 | NonFraud=0.0031
                Q75 bayes_risk    | Fraud=0.7484 | NonFraud=0.0111
                KS = 0.748, p-value = 0.00e+00
                AUC (Bayes-only) = 0.942 

            >>> Conclusion:
                - Fraud cao hơn ~100x so với non-fraud → Separation rất rõ
                - KS_stats = 0.748 => Đây là strong prior, gần như “manual model”
                - AUC cao; Bayes score đủ mạnh để làm rule engine độc lập, hoặc làm feature trụ cột cho ML nhưng cần chú ý các chỉ số cho nhóm fraud 
                (sẽ bàn tiếp trong phần confusion matrix)            
    """
    # Initialize Bayesian Risk 
    profiler = BayesianRiskProfiler(min_count=50)

    profiler.fit(
        cur_df_trunc,
        categorical_cols=['Payment', 'Channel', 'SiteName']
    )

    # 1. Inspect insight (EDA)
    print("Inspect insight (EDA)")
    display(profiler.tables['SiteName'].head(10))
    display(profiler.tables['Payment'].head(10) )

    # 2. Metric Evaluation
    cur_df_trunc['bayes_risk'] = profiler.score(
        cur_df_trunc,
        weights={
            'SiteName': 1.5,
            'Payment': 1.0,
            'Channel': 0.5
        }
    )

    # Distribution separation (fraud vs non-fraud)
    fraud = cur_df_trunc[cur_df_trunc['Label'] == 1]['bayes_risk']
    nonfraud = cur_df_trunc[cur_df_trunc['Label'] == 0]['bayes_risk']

    print(f"Median bayes_risk | Fraud={fraud.median():.4f} | NonFraud={nonfraud.median():.4f}")
    print(f"Q75 bayes_risk   | Fraud={fraud.quantile(0.75):.4f} | NonFraud={nonfraud.quantile(0.75):.4f}")

    # Compute 
    ks = ks_2samp(fraud, nonfraud)
    print(f"KS = {ks.statistic:.3f}, p-value = {ks.pvalue:.2e}")

    # Compute AUC
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(cur_df_trunc['Label'], cur_df_trunc['bayes_risk'])
    print(f"AUC (Bayes-only) = {auc:.3f}")

    return cur_df_trunc, 

# ========================================= 14. Threshold Selection ========================================== #
def get_table_threshold_wrt_othermetrics(cur_df_trunc, thresh_val = 0.03):
    """
        Args & usage:
            - cur_df_trunc : updated current-truncated-data after using `get_eda_results`
                >>> cur_df_trunc = get_eda_result(cur_df_trunc)
            - thresh_val : threshold value

        Results:
            =============================================================================================================
            1. Table of thresholds
            ------------------------------------------------------------------------------------------------------------
            Table threshold:

                |    idx    | threshold | precision | recall   | alert_rate | lift      |
                |-----------|-----------|----------|------------|-----------|-----------|
                | 11081     | 0.996607  | 1.000000 | 0.007601   | 0.000092  | 83.017736 |
                | 11080     | 0.989329  | 0.900000 | 0.007601   | 0.000102  | 74.715963 |
                | 11079     | 0.979549  | 0.909091 | 0.008446   | 0.000112  | 75.470670 |
                | 11078     | 0.971063  | 0.916667 | 0.009291   | 0.000122  | 76.099592 |
                | 11077     | 0.965250  | 0.991525 | 0.098818   | 0.001200  | 82.314196 |
                | 11076     | 0.957452  | 0.991935 | 0.103885   | 0.001262  | 82.348239 |
                | 11075     | 0.955603  | 0.862069 | 0.105574   | 0.001475  | 71.567014 |
                | 11074     | 0.951838  | 0.890110 | 0.136824   | 0.001852  | 73.894908 |
                | 11073     | 0.943746  | 0.885246 | 0.136824   | 0.001862  | 73.491111 |
                | 11072     | 0.941666  | 0.835052 | 0.136824   | 0.001974  | 69.324089 |

            >> Rule engine (high precision):

                |    idx    | threshold | precision | recall   | alert_rate |   lift    |
                |-----------|-----------|----------|------------|-----------|-----------|
                | 10551     | 0.215023  | 0.200746 | 0.591216   | 0.035476  | 16.665448 |
                | 10552     | 0.215087  | 0.200803 | 0.591216   | 0.035465  | 16.670228 |
                | 10553     | 0.215487  | 0.203725 | 0.591216   | 0.034957  | 16.912810 |
                | 10554     | 0.216414  | 0.203844 | 0.591216   | 0.034936  | 16.922660 |
                | 10555     | 0.216625  | 0.203903 | 0.591216   | 0.034926  | 16.927590 |

            >> Early warning / triage:

                |  idx  | threshold | precision | recall   | alert_rate | lift     |
                |-------|-----------|-----------|----------|------------|----------|
                | 0     | 0.000019  | 0.012046  | 1.0      | 1.000000   | 1.000000 |
                | 1     | 0.000020  | 0.012047  | 1.0      | 0.999919   | 1.000081 |
                | 2     | 0.000020  | 0.012048  | 1.0      | 0.999807   | 1.000193 |
                | 3     | 0.000021  | 0.012051  | 1.0      | 0.999583   | 1.000417 |
                | 4     | 0.000021  | 0.012054  | 1.0      | 0.999298   | 1.000702 |

            >> Review capacity (x% transaction):

                |  idx  | threshold | precision | recall   | alert_rate | lift     |
                |-------|-----------|-----------|----------|------------|----------|
                | 10379 | 0.144450  | 0.157948  | 0.655405 | 0.049983   | 13.11251 |
                | 10380 | 0.144674  | 0.157980  | 0.655405 | 0.049973   | 13.11518 |
                | 10381 | 0.144978  | 0.158045  | 0.655405 | 0.049953   | 13.12052 |
                | 10382 | 0.145044  | 0.158077  | 0.655405 | 0.049943   | 13.12320 |
                | 10383 | 0.145245  | 0.157766  | 0.653716 | 0.049912   | 13.09738 |

            >> confusion matrix:

                    [[ 83492     13617 ]
                     [  140      1044 ]]

            >>> Conclusion:
                - High precision mode (rule engine); dùng khi: muốn bắt fraud “chắc tay”, chấp nhận miss nhiều
                - Confusion matrix:

                    👉 FN rất thấp → ít miss fraud nặng                
                    👉 FP nhiều → phù hợp review, không phải auto-block

            =============================================================================================================
            2. Sanity check: 
            -------------------------------------------------------------------------------------------------------------
                risk_bin
                (-0.0009809, 0.000508]    0.000000
                (0.000508, 0.000944]      0.000000
                (0.000944, 0.00147]       0.000103
                (0.00147, 0.00211]        0.000808
                (0.00211, 0.00316]        0.000718
                (0.00316, 0.00465]        0.000914
                (0.00465, 0.00811]        0.001121
                (0.00811, 0.0175]         0.005479
                (0.0175, 0.0597]          0.016628
                (0.0597, 0.997]           0.094758
                Name: Label, dtype: float64

                Top 1% captures 34.15x fraud
                Top 2% captures 22.65x fraud
                Top 5% captures 13.11x fraud

            >>> Conclusion:
                - Lowest bin = ~0% and top bin (last decile): ~9.5% => Fraud rate tăng đơn điệu theo bayes_risk; 
                                                                       Score calibrated theo thứ tự, không bị đảo chiều
                - Top-K capture: cao nếu so với fraud baseline ~1.2%
    """

    y = cur_df_trunc['Label']
    s = cur_df_trunc['bayes_risk']

    # Precision–Recall curve
    precision, recall, thresholds = precision_recall_curve(y, s)
    pr_table = pd.DataFrame({
        'threshold': thresholds,
        'precision': precision[:-1],
        'recall': recall[:-1]
    })

    # deploy-oriented: thêm capture rate & alert rate
    pr_table['alert_rate'] = (s.values[:, None] >= thresholds).mean(axis=0)
    pr_table['lift'] = pr_table['precision'] / y.mean()
    print("Table threshold:")
    pr_table.sort_values('threshold', ascending=False).head(10)

    # Select threshold with respect to (wrt) business objective
    print("Rule engine (high precision):")
    display(pr_table[pr_table['precision'] > 0.2].head())

    print("Early warning / triage:")
    display(pr_table[pr_table['recall'] > 0.3].head())

    print("Review capacity (x% transaction):")
    pr_table.loc[
        (pr_table['alert_rate'] < 0.05) & (pr_table['lift'] > 3)
    ].head()

    # confusion matrix 
    y_pred = (cur_df_trunc['bayes_risk'] >= thresh_val).astype(int)

    cm = confusion_matrix(cur_df_trunc['Label'], y_pred)
    print("confusion matrix:")
    print(cm)

    # Calibration check (Bayes có bị overconfident?)
    cur_df_trunc['risk_bin'] = pd.qcut(
        cur_df_trunc['bayes_risk'], 10, duplicates='drop'
    )

    calib = cur_df_trunc.groupby('risk_bin')['Label'].mean()
    print(calib)

    for k in [0.01, 0.02, 0.05]:
        topk = cur_df_trunc.nlargest(int(len(cur_df_trunc)*k), 'bayes_risk')
        print(f"Top {int(k*100)}% captures {topk['Label'].mean() / y.mean():.2f}x fraud")
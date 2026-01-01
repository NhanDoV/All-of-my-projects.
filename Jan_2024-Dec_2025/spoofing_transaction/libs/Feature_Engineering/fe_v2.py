import numpy as np
import pandas as pd
from scipy.special import logit, expit

class BayesianEncoder:
    def __init__(
        self,
        alpha=0.5, beta=0.5,      # Jeffreys prior
        k=100,                    # reliability constant
        min_count=10,
        p_floor=1e-4,
        p_ceil=1-1e-4,
        cap_logit=4.0
    ):
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.min_count = min_count
        self.p_floor = p_floor
        self.p_ceil = p_ceil
        self.cap_logit = cap_logit

    def fit(self, df, col, label_col='Label'):
        g = df.groupby(col)[label_col].agg(['count', 'sum'])
        g['p_raw'] = (g['sum'] + self.alpha) / (g['count'] + self.alpha + self.beta)

        self.global_prior = df[label_col].mean()

        g['w'] = g['count'] / (g['count'] + self.k)
        g['p_smooth'] = g['w'] * g['p_raw'] + (1 - g['w']) * self.global_prior

        g['p_smooth'] = g['p_smooth'].clip(self.p_floor, self.p_ceil)
        g['logit'] = logit(g['p_smooth']).clip(-self.cap_logit, self.cap_logit)

        self.table = g
        self.col = col
        return self

    def transform(self, df):
        out = df[[self.col]].merge(
            self.table[['logit']],
            left_on=self.col,
            right_index=True,
            how='left'
        )
        return out['logit'].fillna(logit(self.global_prior))

# Numeric bucketing (Amount)
def bucket_amount(x):
    if x < 0:
        return 'neg'
    elif x <= 10:
        return '0_10'
    elif x <= 50:
        return '10_50'
    elif x <= 200:
        return '50_200'
    else:
        return '200_plus'

# Rule flags (weak signals)
def build_rule_flags(df):
    return pd.DataFrame({
        'is_cash_refund_25': (
            (df['Payment'].str.contains('CASH', na=False)) &
            (df['Total_amount_abs'] > 25)
        ).astype(int),

        'is_high_channel': (df['Channel'] >= 25).astype(int),

        'is_exception_sc': df['exception_type'].str.contains('SC', na=False).astype(int)
    })


# Bayesian pipeline
def build_bayes_features(df):
    df = df.copy()

    # -------- Global prior --------
    p0 = df['Label'].mean()
    base_logit = logit(p0)

    # -------- Amount bucket --------
    df['amt_bucket'] = df['Total_amount_abs'].apply(bucket_amount)

    # -------- Fit encoders --------
    enc_site = BayesianEncoder(k=100).fit(df, 'SiteName')
    enc_channel = BayesianEncoder(k=200).fit(df, 'Channel')
    enc_payment = BayesianEncoder(k=100).fit(df, 'Payment')
    enc_amt = BayesianEncoder(k=50).fit(df, 'amt_bucket')

    # -------- Transform --------
    df['logit_site'] = enc_site.transform(df)
    df['logit_channel'] = enc_channel.transform(df)
    df['logit_payment'] = enc_payment.transform(df)
    df['logit_amt'] = enc_amt.transform(df)

    # -------- Rules --------
    rules = build_rule_flags(df)

    # -------- Final score --------
    df['bayes_raw_score'] = (
        base_logit
        + df['logit_site']
        + df['logit_channel']
        + df['logit_payment']
        + df['logit_amt']
        + 0.3 * rules['is_cash_refund_25']
        + 0.2 * rules['is_high_channel']
        + 0.1 * rules['is_exception_sc']
    )

    df['bayes_score'] = expit(df['bayes_raw_score'])

    return df



"""
    Sau khi thực hiện eda_v2; ta có các quyết định sau (bên cạnh việc drop data từ 2023 về trước):
        BAYES SCORE – FEATURE ENGINEERING SUMMARY
        ========================================

        GOAL
        ----
        Build a standalone Bayesian risk scoring engine for spoofing transactions.
        The score is used for ranking and tier-based actions (Tier B = 1% manual review).
        No ML, no rolling window, no auto-adaptation.

        Core principles:
        - Stability > sensitivity
        - Ranking > calibration
        - No hard decision by AI
        - Explainable, auditable, low leakage risk


        DATA WINDOW
        -----------
        - Training window (frozen): Jan-2024 → Dec-2024
        - Only finalized fraud labels are used
        - No future data, no rolling update


        FEATURE GROUPS
        --------------

        1. Global Prior
        ---------------
        - Global fraud rate over training window
        - Used as anchor for all Bayes computations
        - Prevents zero-risk or absolute-risk behavior


        2. Core Categorical Features (Bayes-applied)
        ---------------------------------------------
        Used directly in Bayesian posterior estimation:

        - SiteName
        - Channel
        - PaymentType
        - (Optional) Region

        NOT used:
        - EmployeeID (high variance, low reliability)
        - Card-level identifiers (no contagion evidence)


        3. Numeric Features (Bucketed, not raw)
        ---------------------------------------
        Numeric values are discretized to avoid overfitting
        and distributional assumptions.

        Examples:
        - Total_amount buckets (e.g. <0, 0–10, 10–50, 50–200, 200+)
        - Exception_amount:
        - Absolute value bucket
        - Signed indicator (refund vs charge)

        Each bucket is treated as a categorical group
        and processed via Bayesian smoothing.


        4. Binary Rule Flags (Weak Signals)
        ----------------------------------
        Used as small additive adjustments only.

        Examples:
        - is_credit_debit
        - is_high_risk_channel
        - is_off_hour
        - cash_refund_25

        Rules never make decisions.
        They only provide marginal score contribution.


        BAYESIAN ESTIMATION
        -------------------

        For each categorical value g:

        - Raw posterior:
        P(fraud | g) = (F_g + α) / (N_g + α + β)

        - Reliability weight:
        w_g = N_g / (N_g + k)

        - Smoothed posterior:
        P*_g = w_g * P(fraud | g) + (1 - w_g) * P0

        Where:
        - P0 = global prior
        - α, β = small priors (e.g. Jeffreys)
        - k = reliability constant (e.g. 50–200)


        LOG-ODDS TRANSFORMATION
        -----------------------
        All probabilities are converted to log-odds:

        - score_g = log( P*_g / (1 - P*_g) )

        Rationale:
        - Additive across features
        - Numerically stable
        - Avoids probability saturation


        SCORE COMPOSITION
        -----------------
        Final raw score is additive:

        RawScore =
            logit(P0)
        + score_SiteName
        + score_Channel
        + score_PaymentType
        + score_AmountBucket
        + Σ (λ_i * rule_flag_i)

        Notes:
        - No manual feature weighting
        - Natural contribution via log-odds
        - Rule flags have small λ (weak influence)


        SAFETY CONTROLS
        ---------------
        - Minimum count threshold:
        If N_g < min_count → ignore feature, use prior

        - Score capping:
        Individual feature contributions are capped
        to prevent domination (e.g. Site risk explosion)

        - Probability floor / ceiling:
        P*_g is constrained to avoid absolute certainty

        - Cold-start handling:
        Unseen categories default to global prior


        FINAL SCORE & RANKING
        --------------------
        - RawScore is passed through sigmoid:
        FinalScore = sigmoid(RawScore)

        - FinalScore is NOT a true probability
        - Used only for ranking and percentile mapping


        TIER MAPPING (FROZEN)
        --------------------
        Percentiles are computed on training distribution
        and frozen for production use.

        - Tier A: top ~0.2% (auto case / urgent)
        - Tier B: next ~1.0% (manual review, primary)
        - Tier C: next ~4% (monitoring)
        - Tier D: rest (allow)

        Site-specific threshold:
        - Optional parameter
        - Default OFF
        - Can be enabled without changing core logic


        OUTPUT FOR BUSINESS / OPS
        -------------------------
        For each Tier B case, provide:

        - FinalScore and percentile
        - Top contributing factors (e.g. Site, Channel, Payment)
        - Historical baseline comparison
        - Clear disclaimer:
        "Risk elevated based on historical patterns;
        final determination requires manual review."


        UPDATE & GOVERNANCE
        -------------------
        - No rolling window
        - No auto-update
        - Quarterly review or triggered recalibration only

        Triggers:
        - Tier B precision < 0.7 (lagged)
        - Combined recall < 0.75
        - Significant lift decay or score drift


        END OF FE SPEC
        ==============
"""
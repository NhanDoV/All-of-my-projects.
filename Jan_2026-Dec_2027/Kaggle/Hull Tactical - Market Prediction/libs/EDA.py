import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import jarque_bera

# JS to hide index column and row headers
hide_index_js = """
<script>
    const tables = window.parent.document.querySelectorAll('table');
    tables.forEach(table => {
        const indexColumn = table.querySelector('thead th:first-child');
        if (indexColumn) {
            indexColumn.style.display = 'none';
        }
        table.querySelectorAll('tbody td, tbody th').forEach(cell => {
            cell.style.fontSize = '10px';
            cell.style.textAlign = 'center';
        });
        const indexCells = table.querySelectorAll('tbody th');
        indexCells.forEach(cell => {
            cell.style.display = 'none';
        });
    });
</script>
"""

def show_plotly_template(fig):
    fig.update_layout(
            plot_bgcolor = 'rgba(7, 10, 245, 0.1)',
            paper_bgcolor = 'rgba(7, 10, 245, 0.71)')
    st.plotly_chart(fig)

def write_card(md_text, kind="info"):
    st.markdown(md_text)
    st.markdown(
        """
        </div>
        """,
        unsafe_allow_html=True
    )

def overview_description(train_df, test_df):
    feature_categories = {
        'Target_Vrs': ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns'],
        'Identifier': ['date_id'],
        'Momentum_Features': [col for col in train_df.columns if col.startswith('MOM')],  # Momentum features
        'Market_Features': [col for col in train_df.columns if (col.startswith('M') & ~col.startswith('MOM'))],
        'Economic_Features': [col for col in train_df.columns if col.startswith('E')],  # Macro Economic features
        'Interest_Features': [col for col in train_df.columns if col.startswith('I')],  # Interest Rate features
        'Price_Features': [col for col in train_df.columns if col.startswith('P')],  # Price/Valuation features
        'Volatility_Features': [col for col in train_df.columns if col.startswith('V')],  # Volatility features
        'Sentiment_Features': [col for col in train_df.columns if col.startswith('S')],  # Sentiment features
        'Dummy_Features': [col for col in train_df.columns if col.startswith('D')],  # Dummy/Binary features
        'Reference_Data': ['price', 'SP500']   
    }
    st.markdown("""
                - Source: [link](https://www.kaggle.com/competitions/hull-tactical-market-prediction/overview)
            """)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("##### Overview dataset")
    with c2:
        sel = st.selectbox("View train or test-set", ["train", "test"])
    if sel == "train":
        st.dataframe(styled_df(train_df.head()), 
                     hide_index=True)
    else:
        st.dataframe(test_df.head(), hide_index=True)

    total_feats = 0
    for _, features in feature_categories.items():
        available = [f for f in features if f in train_df.columns]
        total_feats = total_feats + len(available)
    df = pd.DataFrame(data = {
                                'category': feature_categories.keys(), 
                                'feats': feature_categories.values()
                            })
    df['n_fea'] = df['feats'].apply(lambda x: len(x))
    df['feats'] = df['feats'].apply(lambda x: ', '.join(x))
    st.write("- Explaination")
    df = df[df['n_fea'] > 0]
    df_styled = styled_df(df[['category', 'n_fea', 'feats']])
    
    st.dataframe(df_styled, hide_index = True, column_config = {})

def styled_df(df):
    return df.style.set_properties(**{
                    "background-color": "#10127C",  # xanh nhạt (blue-50)
                    "color": "#D1DBE8",             # xanh đậm (blue-700)
                    "font-weight": "500",
                    "border": "1px solid #BBDEFB",  # viền xanh nhạt
                    "padding": "8px",
                }).set_table_styles([
                    {"selector": "th", "props": [
                        ("background-color", "#09528D"),  # header xanh dương chuẩn
                        ("color", "white"),
                        ("font-weight", "bold"),
                    ]}
                ])#.format(precision=0, na_rep="-")

def get_metric_overviews(train_df, test_df):

    def metric_card(label, value):
        return f"""
        <div class="metric-card metric-center">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """

    common_cols = set(train_df.columns).intersection(set(test_df.columns))
    n_common_cols = metric_card("n-common-cols", len(common_cols))
    train_dim = metric_card("Train set dim", str(train_df.shape))
    test_dim = metric_card("Test set dim", test_df.shape)
    time_gap = metric_card("Time gap", test_df['date_id'].min() - train_df['date_id'].max())
 
    st.markdown(f"""
        <div class="metric-table-grid">
            {n_common_cols}
            {train_dim}
            {test_dim}
            {time_gap}
        </div>
    """, unsafe_allow_html=True)

    with st.expander("See more", expanded = True):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.write("- Cols in train, NOT in test : ", set(train_df.columns) - set(test_df.columns))
            st.write("- Cols in test, NOT in train : ", set(test_df.columns) - set(train_df.columns))
        with c2:
            st.write(f"Train period: `date_id` {train_df['date_id'].min()} to {train_df['date_id'].max()}")
            st.write(f"Test period:     `date_id` {test_df['date_id'].min()} to {test_df['date_id'].max()}")

def get_dtype_distribution(train_df, test_df):

    ser_train = train_df.dtypes.value_counts().reset_index()
    ser_train.columns = ['dtype', 'count']
    ser_train['note'] = 'train-set'
    ser_test = test_df.dtypes.value_counts().reset_index()
    ser_test.columns = ['dtype', 'count']
    ser_test['note'] = 'test-set'
    res = pd.concat([ser_train, ser_test]).sort_values(by = 'count')
    res['dtype'] = res['dtype'].astype(str)

    fig = px.bar(res, y='dtype', x='count', 
                color='note', barmode='group', 
                color_discrete_map = {'train-set': "#229D1B", 'test-set': "#AD442F"},
                text = 'count', height = 333)

    fig.update_yaxes(ticklabelstandoff = 10)
    fig.update_xaxes(ticklabelstandoff = 5)

    with st.expander("See chart", expanded = True):
        show_plotly_template(fig)

# ======================= Main target analytic : forward_returns ==================================
def get_general_stats(train_df, target_col = 'forward_returns'):
    target_ser = train_df[target_col]

    avg = target_ser.mean()
    sd = target_ser.std()
    skw = target_ser.skew()
    kur = target_ser.kurtosis()
    
    stats_df = pd.DataFrame({
        'stats': ['AVG', 'std', 'skewness', 'kurtosis'],
        'value': [avg, sd, skw, kur]
    })
    
    st.write("- `Table stats`")
    st.components.v1.html(hide_index_js, height = 0)
    st.table(stats_df) # st.dataframe(stats_df, hide_index = True)

def get_percentile_table(train_df, target_col = 'forward_returns'):
    target_ser = train_df[target_col]
    st.write("- `Percentile table`")
    indexes = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentile_df = pd.DataFrame({
        'percentiles': [f"{idx}th" for idx in indexes ],
        'values': [np.percentile(target_ser, idx) for idx in indexes]
        }).round(5)
    st.components.v1.html(hide_index_js, height=0)
    st.table(percentile_df)

def get_table_risks(train_df, target_col):
    target_ser = train_df[target_col]
    st.write("- `Table of risks`")
    df = pd.DataFrame(
            {
                'Daily \n Volatility': [target_ser.std()],
                'Annualized \n Volatility': [target_ser.std()*np.sqrt(252)],
                'Value at \n Risk (95%)' : [np.percentile(target_ser, 5)],
                'Value at \n Risk (99%)' : [np.percentile(target_ser, 1)]
            }, index = [0]
        )
    st.components.v1.html(hide_index_js, height = 0)
    st.table(df)

def get_normality_test(train_df, target_col):
    target_ser = train_df[target_col]
    stat_jb, p_jb = jarque_bera(target_ser)
    st.write("- `Normality testing results:`")
    c1, c2, c3 = st.columns([1.5, 1, 4])
    
    TITLE_COLOR = "#2E7D32"
    with c1:
        st.write(f"""
                 <div style="text-align:center; font-size:20px; color: {TITLE_COLOR}">
                    <b> Test-statistic </b>
                 </div>
                 <div style="text-align:center; font-size:15px;">
                    <b>{stat_jb:.4f}</b>
                 </div>
                 """, unsafe_allow_html = True)

    with c2:
        st.write(f"""
                 <div style="text-align:center; font-size:20px; color: {TITLE_COLOR}">
                    <b> p.value  </b>
                 </div>
                 <br>
                 <div style="text-align:center; font-size:15px;">
                    <b>{p_jb:.6f}</b>
                 </div>
                 """, unsafe_allow_html = True)

    with c3:
        st.write(f"""
                 <div style="text-align:center; font-size:20px; color: {TITLE_COLOR}">
                    <b> Conclusion </b>
                 </div>
                 <div style="text-align:center; font-size:15px;">
                    <b>{'No <br> (returns are NOT normally distributed)' if p_jb < 0.05 else 'Yes'}</b>
                 </div>
                 """, unsafe_allow_html = True)

def get_main_target_analytic(train_df, target_col = 'forward_returns'):
    c1, _, c2 = st.columns([2.2, 0.01, 5])
    with c1:
        with st.expander("General info", expanded = True):
            c1a, _, c1b = st.columns([1, 0.1, 1])
            with c1a:
                get_general_stats(train_df, target_col)
            with c1b:
                get_percentile_table(train_df, target_col)
            get_table_risks(train_df, target_col)
            get_normality_test(train_df, target_col)

            st.write("----------")
            wsize = st.number_input("Select window-size", min_value=20, max_value=100)

    with c2:
        with st.expander("Chart", expanded = True):
            boxplot_and_distribution(train_df, target_col)
            get_QQplot_and_timeseries(train_df, target_col)
            get_rolling_stats(train_df, target_col, window = wsize)

def count_null(df, note="Train", low_thresh=10, high_thresh=70):
    
    null_cnt = df.isnull().sum()
    null_cnt = null_cnt[null_cnt > 0].sort_values(ascending=False).reset_index()
    
    if len(null_cnt) == 0:
        print(f"{note} dataset does not have missing data")
        return

    null_cnt.columns = ['col', 'count']
    null_cnt['perc'] = (null_cnt['count'] / len(df) * 100).round(2)

    fig = px.bar(null_cnt, x = 'col', y = 'perc', 
                 hover_data = ['count'], height = 500,
                 color='perc', color_continuous_scale = px.colors.sequential.Greens, 
                 title = "Missing Data Percentage")

    # Threshold lines
    fig.add_hline(y = low_thresh, line_dash = "dash", line_color =  "#e0af96",
                  annotation_text="Low threshold (10%)", annotation_position="top right")

    fig.add_hline(y=high_thresh, line_dash="dash", line_color = "#e45133",
                  annotation_text="High threshold (70%)", annotation_position="top right")

    fig.update_layout(yaxis_title = "Missing Percentage (%)", xaxis_title = "Columns", 
                      coloraxis_colorbar = dict(
                                                title = "%_Missing.of.colorbar", orientation = "h",
                                                y = 1.05, x = 0.5,
                                                yanchor = "bottom", xanchor = "center"
                                            ),
                      )
    fig.update_xaxes(tickangle = -90, ticklabelstandoff = 10)
    show_plotly_template(fig)

def get_top_missing_val_wrt_cols(train_df, top_cols = 5):

    missing_stats = pd.DataFrame({
        'Missing_Count': train_df.isnull().sum(),
        'Percentage': (train_df.isnull().sum() / len(train_df)) * 100,
        'First_Valid_Index': train_df.apply(lambda x: x.first_valid_index()),
        'Last_Valid_Index': train_df.apply(lambda x: x.last_valid_index())
    }).sort_values('Percentage', ascending=False)

    st.write(f"Top {top_cols} Columns with Missing Values")
    st.dataframe(styled_df(missing_stats[missing_stats['Missing_Count'] > 0].head(top_cols)) )

def get_missing_data_report(train_df, test_df):
    c1, _, c2 = st.columns([2, 0.01, 5])
    with c1:
        top_cols = st.number_input("Select top-columns", min_value = 5, max_value = 90)
        get_top_missing_val_wrt_cols(train_df, top_cols = top_cols)
    with c2:
        with st.expander("Chart", expanded = True):
            count_null(train_df, test_df)


def get_rolling_stats(train_df, target_col, window = 50):
    target_ser = train_df[target_col]
    
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize = (20, 4.5))
    plt.rcParams.update({
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11
    })

    # Rolling statistics
    ax[0].plot(train_df['date_id'], target_ser, label='Actual', alpha=0.3)
    ax[0].plot(train_df['date_id'], target_ser.rolling(window=window).mean(), 
                    label=f'{window}-period MA', linewidth=2)
    ax[0].plot(train_df['date_id'], target_ser.rolling(window=window).std(), 
                    label=f'{window}-period Std', linewidth=2)
    ax[0].set_title(f'Rolling Statistics (window.size = {window})')
    ax[0].set_xlabel('Date ID')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Cumulative returns
    ax[1].plot(train_df['date_id'], (1 + target_ser).cumprod() - 1)
    ax[1].set_title('Cumulative Returns')
    ax[1].set_xlabel('Date ID')
    ax[1].set_ylabel('Cumulative Return')
    ax[1].grid(True, alpha=0.3)

    st.pyplot(fig)

def get_QQplot_and_timeseries(train_df, target_col):
    target_ser = train_df[target_col]
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize = (20, 4.5))
    plt.rcParams.update({
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11
    })
    # Q-Q plot
    stats.probplot(target_ser.dropna(), dist="norm", plot = ax[0])
    ax[0].set_title('Q-Q Plot (Normality Test)')

    # Time series
    ax[1].plot(train_df['date_id'], target_ser, linewidth=0.8, alpha=0.7)
    ax[1].axhline(0, color='red', linestyle='--', alpha=0.5)
    ax[1].set_title('Forward Returns Over Time')
    ax[1].set_xlabel('Date ID')
    ax[1].set_ylabel('Returns')
    ax[1].grid(True, alpha=0.3)

    st.pyplot(fig)

def boxplot_and_distribution(train_df, target_col):
    target_ser = train_df[target_col]
    returns = target_ser.dropna()
    c1, _, c2, _, c3, _, c4, _, c5 = st.columns([1, 0.1, 1, 0.1, 1, 0.1, 1, 0.1, 1])
    with c1:
        write_card(
            f"""
            `Negative Returns`
            - Count: **{(returns < 0).sum()}**
            - Ratio: `{(returns < 0).sum()/len(returns)*100:.2f}%`
            """,
            kind="warning"
        )

    with c2:
        write_card(
            f"""
            `Zero Returns`
            - Count: **{(returns == 0).sum()}**
            - Ratio: `{(returns == 0).sum()/len(returns)*100:.2f}%`
            """,
            kind="info"
        )

    with c3:
        write_card(
            f"""
            `Positive Returns`
            - Count: **{(returns > 0).sum()}**
            - Ratio: `{(returns > 0).sum()/len(returns)*100:.2f}%`
            """,
            kind="success"
        )

    with c4:
        write_card(
            f"""
            `Annualized Volatility`
            - Value: `{returns.std() * np.sqrt(252):.4f}`
            """,
            kind="info"
        )

    with c5:
        excess_returns = returns - train_df['risk_free_rate'].dropna().mean()
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

        write_card(
            f"""
            `Sharpe Ratio`
            - Value: `{sharpe:.4f}`
            """,
            kind="success"
        )

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize = (20, 4.5))

    plt.rcParams.update({
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11
    })

    # histogram
    ax[0].hist(target_ser.dropna(), bins=100, edgecolor = 'black', alpha = 0.7)
    ax[0].axvline(target_ser.mean(), color='red', linestyle='--', 
                  label = f'Mean: {target_ser.mean():.6f}')
    ax[0].axvline(target_ser.median(), color='green', linestyle='--', 
                  label = f'Median: {target_ser.median():.6f}')
    ax[0].set_title('Forward Returns Distribution')
    ax[0].set_xlabel('Returns')
    ax[0].set_ylabel('Frequency')
    ax[0].legend()

    # boxplot
    sns.boxplot(target_ser.dropna(), orient = "h", ax = ax[1])
    ax[1].set_title('Forward Returns - Box Plot')
    ax[1].set_ylabel('Returns')

    # display in streamlit
    st.pyplot(fig)

def get_correlative_features_analysis(train_df, test_df):

    # Auto define common columns
    common_cols = set(train_df.columns).intersection(set(test_df.columns))

    # Cluster specific columns by group
    col_clusters = {
        'Market_Features': [col for col in train_df.columns if (col.startswith('M') & ~col.startswith('MOM'))],
        'Economic_Features': [col for col in train_df.columns if col.startswith('E')],  # Macro Economic features
        'Interest_Features': [col for col in train_df.columns if col.startswith('I')],  # Interest Rate features
        'Price_Features': [col for col in train_df.columns if col.startswith('P')],  # Price/Valuation features
        'Volatility_Features': [col for col in train_df.columns if col.startswith('V')],  # Volatility features
        'Sentiment_Features': [col for col in train_df.columns if col.startswith('S')],  # Sentiment features
        'Dummy_Features': [col for col in train_df.columns if col.startswith('D')],  # Dummy/Binary features
    }

    # some supported functions
    def get_corrcoef_table(train_df, target_col):
        corr_mat = train_df.corr()
        corr_ser = corr_mat[target_col]
        selected_index = corr_ser.index.intersection(common_cols)
        corr_ser = corr_ser[selected_index]
        pos_rel = corr_ser[corr_ser > 0].sort_values()
        neg_rel = corr_ser[corr_ser < 0].sort_values(ascending = False)

        return pos_rel, neg_rel
    
    def corr_coef_plot(pos_rel, neg_rel, target_col, title = "Whole Feature Correlation"):
        max_height = min(900, 45*max(len(pos_rel), len(neg_rel)) )
        fig = make_subplots(
            rows = 1, cols = 2,
            column_widths = [0.5, 0.5],
            subplot_titles=("Positive Correlation", "Negative Correlation")
        )

        # Positive correlations (left)
        fig.add_trace(
            go.Bar(x = pos_rel.values, y = pos_rel.index,
                orientation="h", name="Positive", marker_color = '#229D1B'),
            row = 1, col = 1
        )

        # Negative correlations (right)
        fig.add_trace(
            go.Bar(x = neg_rel.values, y = neg_rel.index,
                orientation = "h", name = "Negative", marker_color = '#AD442F'),
            row = 1, col = 2
        )

        fig.update_layout(
                            height = max_height, showlegend = False,
                            title = f"{title} with {target_col.capitalize()}",
                        )

        # update axes
        fig.update_xaxes(title="Correlation")
        fig.update_yaxes(side="right", ticklabelstandoff=15, row=1, col=2)
        fig.update_yaxes(title="Feature", side="left", ticklabelstandoff=15, row=1, col=1)

        return fig

    def specific_group_corrcoef(train_df, target_col, sel_group):
        sel_cols = col_clusters[sel_group]
        pos_rel, neg_rel = get_corrcoef_table(train_df, target_col)
        st.write(f"Result")
        pos_rel = pos_rel[pos_rel.index.isin(sel_cols)]
        neg_rel = neg_rel[neg_rel.index.isin(sel_cols)]
        fig = corr_coef_plot(pos_rel, neg_rel, target_col)
        show_plotly_template(fig)

    def advanced_analysis_corrcoef(train_df, prefix, corrcoef_thresh = 0.8):
        c1, _, c2 = st.columns([5, 0.05, 2])
        
        cols = [col for col in train_df.columns if col.startswith(prefix)]
        
        if len(cols) > 1:
            # Correlation within group
            group_corr = train_df[cols].corr()
            
            # Plot heatmap
            with c1:
                fig, ax = plt.subplots(figsize=(10, 9))
                hm = sns.heatmap(group_corr, cmap = 'RdYlGn', 
                            center = 0, ax = ax, 
                            annot = True, fmt = ".2f", annot_kws = {"size": 6},
                            square = True, linewidths = 0.5, 
                            cbar_kws = {
                                "shrink": 0.8,
                                "pad": 0.08,
                                "orientation": "horizontal"
                                }
                                )
                cbar = hm.collections[0].colorbar
                cbar.ax.tick_params(labelsize = 6)
                cbar.ax.xaxis.set_ticks_position('top')
                cbar.ax.xaxis.set_label_position('top')

                ax.tick_params(axis='x', labelsize = 8)
                ax.tick_params(axis='y', labelsize = 8)

                plt.title(f'Correlation Matrix: {prefix} Features')
                plt.tight_layout()
                st.pyplot(fig)
            
            # Find highly correlated pairs
            corr_pairs = []
            for i in range(len(group_corr.columns)):
                for j in range(i + 1, len(group_corr.columns)):
                    if abs(group_corr.iloc[i, j]) > corrcoef_thresh:
                        corr_pairs.append({
                            'Feature1': group_corr.columns[i],
                            'Feature2': group_corr.columns[j],
                            'CorrCoef': group_corr.iloc[i, j]
                        })
            
            if corr_pairs:
                with c2:
                    st.write(f"##### Highly correlated pairs in `{prefix} features`")
                    st.write("( `|r| > 0.8` )")
                    st.dataframe(styled_df(pd.DataFrame(corr_pairs)), hide_index = True)

    r3a, _, r3b = st.columns([1, 0.1, 6])
    with r3a:
        target_col = st.selectbox('select the target-column', 
                                ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns'])
        view_mode = st.selectbox("view mode",
                                 ["whole-correlation-coefficients analysis", "specific-group", "Advanced correlation analysis"] )
        if view_mode == "whole-correlation-coefficients analysis":
            with r3b:
                    pos_rel, neg_rel = get_corrcoef_table(train_df, target_col)
                    fig = corr_coef_plot(pos_rel, neg_rel, target_col)
                    show_plotly_template(fig)
        elif view_mode == "specific-group":
            sel_group = st.selectbox("Select group", col_clusters.keys())
            with r3b:
                specific_group_corrcoef(train_df, target_col, sel_group)
        else:
            pref = st.selectbox("Select prefix (group)", 
                                ['E', 'M', 'P', 'S', 'V'], 
                                help = "First letter in each group of features, e.g, `E` meant `Economic-features`, `M`: momentum")
            corrcoef_thresh = st.number_input("Corr-coef thresh.val",
                                              min_value = 0.6, max_value = 0.8)
            with r3b:
                advanced_analysis_corrcoef(train_df, prefix = pref, corrcoef_thresh = corrcoef_thresh)

def dummy_feat_report(train_df, test_df):
    d_features = [col for col in train_df.columns if col.startswith('D')]

    def get_plots(train_df):

        # Define some plotting function 
        my_palette = {
            -1: "#D86228",
            0: "#7ABABC",
            1: "#12086B",
        }

        # 1. Bar plot - agg by a certain function
        def all_barplot(train_df, groupby):
            fig, ax = plt.subplots(3, 3, figsize=(16, 8.8))
            ax = ax.ravel()
            for idx, col in enumerate(d_features):
                if idx < 9:

                    if groupby == "Average":
                        ser = train_df.groupby(col)['forward_returns'].mean()
                    elif groupby == "Median":
                        ser = train_df.groupby(col)['forward_returns'].median()
                    else:
                        ser = train_df.groupby(col)['forward_returns'].std()                    
                    
                    colors = [my_palette.get(v, "#999999") for v in ser.index]

                    ser.plot(kind="bar", ax = ax[idx], color=colors)
                    ax[idx].set_title(f'{col}', fontsize=10, fontweight='bold')
                    ax[idx].set_xlabel('value')
                    ax[idx].grid(alpha=0.3, axis='y')
                    ax[idx].axhline(0, color='red', linestyle='--', linewidth = 1)

            plt.tight_layout()
            st.pyplot(fig)

        # 2. Boxplot
        def all_boxplot(train_df):
            fig, ax = plt.subplots(3, 3, figsize=(16, 8.8))
            ax = ax.ravel()
            for idx, col in enumerate(d_features):
                if idx < 9:
                    sns.boxplot(train_df, y = 'forward_returns', hue = col, gap = .2,
                                medianprops = {"color": "r", "linewidth": 2},
                                palette = my_palette,
                                ax = ax[idx])
                    ax[idx].set_title(f'{col}', fontsize = 10, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

        # 3. Displot 
        def all_displot(train_df):
            fig, ax = plt.subplots(3, 3, figsize=(16, 8.8))
            ax = ax.ravel()
            for idx, col in enumerate(d_features):
                if idx < 9:
                    sns.kdeplot(train_df, x = 'forward_returns', hue = col,
                                palette = my_palette, 
                                multiple = "fill", common_norm = True,
                                ax = ax[idx])
                    ax[idx].set_title(f'{col}', fontsize = 10, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

        # 4. Percentage of bar
        def all_percentage(train_df, test_df):

            fig, ax = plt.subplots(3, 3, figsize = (16, 8.8))
            ax = ax.ravel()

            for idx in range(9):

                d_col = f"D{idx + 1}"
                ser_train = (100 * train_df.groupby(d_col).size() / len(train_df)).reset_index()
                ser_test = (100 * test_df.groupby(d_col).size() / len(test_df)).reset_index()
                
                ser_train['note'] = 'train-set'
                ser_test['note'] = 'test-set'
                
                temp_df = pd.concat([ser_train, ser_test]).reset_index(drop = True)
                temp_df.columns = ['value', 'percentage', 'note']

                sns.barplot(temp_df, x = 'value', y = 'percentage', 
                            palette = {'train-set': "#0C6108", 'test-set': '#EE4B2B'},
                            hue = 'note', ax = ax[idx])
                ax[idx].set_xlabel(d_col)
            
            plt.tight_layout()
            st.pyplot(fig)            

        # 5. Specific dummy-feature : included boxplot plotly (mean - quantiles) & pie-chart on train-set
        def specific_dcol(train_df, d_col):
            c1, _, c2 = st.columns([2, 0.1, 2])
            with c1:
                fig1 = px.box(train_df, x = d_col, color = d_col, height = 500,
                             y = 'forward_returns', color_discrete_map = my_palette)
                fig1.update_layout(
                        plot_bgcolor = 'rgba(8, 13, 255, 0.09)',
                        paper_bgcolor = 'rgba(29, 47, 215, 0.8)')
                show_plotly_template(fig1)
            with c2:
                pie_df = (100 * train_df.groupby(d_col).size() / len(train_df)).reset_index()
                pie_df.columns = [d_col, 'count']
                fig2 = px.pie(pie_df, values = 'count', 
                              names = d_col, height = 500,
                              color_discrete_map = my_palette
                              )
                show_plotly_template(fig2)

        # Layout of plot
        c1, c2, c3 = st.columns(3)
        with c1:
            chart_view = st.selectbox("Select chart-view", 
                                    ["all-barplot", "all-boxplot", "all-distplot", 
                                    "compare percentage both dataset", "specific dummy feature"])
        if chart_view == "all-barplot":
            with c2:
                groupby = st.selectbox("aggfunc", ["Average", "Median", "Standard Deviation"])   
            with c3:
                st.write(" ")
                st.write(f"#### {groupby} Forward Returns")         
            all_barplot(train_df, groupby)
        elif chart_view == "all-boxplot":
            all_boxplot(train_df)
        elif chart_view == "all-distplot":
            all_displot(train_df)
        elif chart_view == "compare percentage both dataset":
            all_percentage(train_df, test_df)
        elif chart_view == "specific dummy feature":
            with c2:
                d_col = st.selectbox("Select dummy features", d_features)
            specific_dcol(train_df, d_col)

    def make_dummy_data_summary(df):
        d_target_analysis = []
        
        for col in d_features:
            for val in df[col].dropna().unique():
                subset_returns = df[df[col] == val]['forward_returns']
                d_target_analysis.append({
                    'Feature': col,
                    'Value': val,
                    'Count': len(subset_returns),
                    'Mean_Return': subset_returns.mean(),
                    'Std_Return': subset_returns.std(),
                    'Median_Return': subset_returns.median()
                })

        d_analysis_df = pd.DataFrame(d_target_analysis)

        return d_analysis_df

    dummy_df_train = make_dummy_data_summary(train_df)

    # Layout of whole report of dummy features
    c1, _, c2 = st.columns([1, 0.01, 2])
    with c1:
        st.write("#### 1. Dummy Features Analytic")
        st.write("- Analyze relationship with target (train-set)")        
        st.components.v1.html(hide_index_js, height = 0)
        st.table(dummy_df_train)

    with c2:
        with st.expander("Chart", expanded = True):
            get_plots(train_df)

def market_and_reference_report(train_df):
    st.write("#### 2. Market & Reference Data Analysis")

    # Some metrics info
    def metric_card(label, value):
        return f"""
        <div class="metric-card metric-center">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """
    
    corr1 = train_df['price'].corr(train_df['forward_returns']).round(5)
    corr2 = train_df['SP500'].corr(train_df['forward_returns']).round(5)
    corr3 = train_df['risk_free_rate'].corr(train_df['forward_returns']).round(5)
    corr4 = train_df['market_forward_excess_returns'].corr(train_df['forward_returns']).round(5)

    st.markdown(f"""
        <div class="metric-table-grid">
            {metric_card("Corr(price, fw_return)", corr1)}
            {metric_card("Corr(SP500, fw_return)", corr2)}
            {metric_card("Corr(risk_free_rate, fw_return)", corr3)}
            {metric_card("Corr(market_forward_excess_returns, fw_return)", corr4)}
        </div>
    """, unsafe_allow_html=True)

    # Analyze SP500 and price relationship
    fig, axes = plt.subplots(2, 3, figsize=(16, 5))
    axes = axes.ravel()

    # SP500 over time
    axes[0].plot(train_df['date_id'], train_df['SP500'], linewidth=1)
    axes[0].set_title('S&P 500 Index Over Time')
    axes[0].set_xlabel('Date ID')
    axes[0].set_ylabel('S&P 500')
    axes[0].grid(True, alpha=0.3)

    # Price over time
    axes[1].plot(train_df['date_id'], train_df['price'], linewidth=1, color='green')
    axes[1].set_title('Price Over Time')
    axes[1].set_xlabel('Date ID')
    axes[1].set_ylabel('Price')
    axes[1].grid(True, alpha=0.3)

    # Scatter: SP500 vs forward returns
    axes[2].scatter(train_df['SP500'], train_df['forward_returns'], alpha=0.3, s=10)
    axes[2].set_title('S&P 500 vs Forward Returns')
    axes[2].set_xlabel('S&P 500')
    axes[2].set_ylabel('Forward Returns')
    axes[2].grid(True, alpha=0.3)

    # Risk-free rate over time
    axes[3].plot(train_df['date_id'], train_df['risk_free_rate'], linewidth=1, color='orange')
    axes[3].set_title('Risk-Free Rate Over Time')
    axes[3].set_xlabel('Date ID')
    axes[3].set_ylabel('Risk-Free Rate')
    axes[3].grid(True, alpha=0.3)

    corr_price_sp500 = train_df['price'].corr(train_df['SP500'])

    # Dual axis comparison
    ax1 = axes[4]
    ax2 = ax1.twinx()
    ax1.plot(train_df['date_id'], train_df['price'], color='green', label='Asset Price', linewidth=1.5)
    ax2.plot(train_df['date_id'], train_df['SP500'], color='blue', label='S&P 500', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('date_id')
    ax1.set_ylabel('Asset Price', color='green')
    ax2.set_ylabel('S&P 500', color='blue')
    ax1.set_title('Asset Price vs S&P 500', fontsize=10, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax2.tick_params(axis='both', labelsize = 8)
    ax2.yaxis.label.set_size(9)

    # Scatter plot
    axes[5].scatter(train_df['SP500'], train_df['price'], alpha=0.5, s=10)
    axes[5].set_title(f'Asset Price vs S&P 500 (Corr: {corr_price_sp500:.4f})', 
                        fontsize=10, fontweight='bold')
    axes[5].set_xlabel('S&P 500')
    axes[5].set_ylabel('Asset Price')
    axes[5].grid(alpha=0.3)

    # Add regression line
    z = np.polyfit(train_df['SP500'].dropna(), train_df['price'].dropna(), 1)
    p = np.poly1d(z)
    axes[5].plot(train_df['SP500'], p(train_df['SP500']), "r--", linewidth=2, alpha=0.8)

    for ax in axes:
        ax.title.set_fontsize(10)
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)
        ax.tick_params(axis='both', labelsize = 8)

    plt.tight_layout()
    st.pyplot(fig)

def feat_avl_report(train_df):
    st.write("#### 3. Feature Availability Analysis")
    feature_categories = {
        'Identifier': ['date_id'],
        'Momentum_Features': [col for col in train_df.columns if col.startswith('MOM')],
        'Market_Features': [col for col in train_df.columns if (col.startswith('M') & ~col.startswith('MOM'))],
        'Economic_Features': [col for col in train_df.columns if col.startswith('E')],
        'Interest_Features': [col for col in train_df.columns if col.startswith('I')],
        'Price_Features': [col for col in train_df.columns if col.startswith('P')],
        'Volatility_Features': [col for col in train_df.columns if col.startswith('V')],
        'Sentiment_Features': [col for col in train_df.columns if col.startswith('S')],
        'Dummy_Features': [col for col in train_df.columns if col.startswith('D')],
        'Target_Variables': ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns'],
        'Reference_Data': ['price', 'SP500']
    }
    availability_info = []

    for category, features in feature_categories.items():
        if category not in ['Identifier', 'Target_Variables', 'Reference_Data', 'Dummy_Features']:
            available = [f for f in features if f in train_df.columns]
            if available:
                # Find first date with non-null data
                first_data = train_df[train_df[available].notna().any(axis=1)]['date_id'].min()
                # Count rows with any data
                rows_with_data = train_df[available].notna().any(axis=1).sum()
                # Calculate completeness
                completeness = (train_df[available].notna().sum().sum() / (len(train_df) * len(available))) * 100
                
                availability_info.append({
                    'Category': category,
                    'Num_Features': len(available),
                    'First_Data_Date': first_data,
                    'Rows_With_Data': rows_with_data,
                    'Overall_Completeness_%': completeness
                })

    availability_df = styled_df(pd.DataFrame(availability_info).sort_values('First_Data_Date'))
    st.dataframe(availability_df, hide_index = True)

def target_relationship_report(train_df):
    
    st.write("#### 5. Excess Returns & Risk-free Rate Analysis")
    c1, _, c2 = st.columns([2, 0.5, 7])
    
    with c1:
        st.write("**Statistic table**")
        risk_ser = train_df['risk_free_rate'].describe()
        mkt_fw_exc_return_ser = train_df['market_forward_excess_returns'].describe()

        result = pd.concat([risk_ser, mkt_fw_exc_return_ser], axis = 1)
        st.dataframe(styled_df(result), hide_index = True)

        st.write("**Other metrics**")
        calculated_excess = train_df['forward_returns'] - train_df['risk_free_rate']
        sharpe_ratio = train_df['market_forward_excess_returns'].mean() / train_df['market_forward_excess_returns'].std() * np.sqrt(252)

        def metric_card(label, value):
            return f"""
            <div class="metric-card metric-center">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """

        st.markdown(f"""
            <div class="metric-table-grid">
                {metric_card("Sharpe_ratio", sharpe_ratio.round(5))}
                {metric_card("Max_difference of market_forward_excess_returns", calculated_excess.max().round(5))}
            </div>
        """, unsafe_allow_html=True)

    with c2:
        c21, c22 = st.columns([7, 3])
        with c21:
            chart = st.selectbox("Select chart", 
                                ["Excess Returns & Risk-free Rate Analysis", 
                                "Verification: market_forward_excess_returns = forward_returns - risk_free_rate"]
                            )            
        if chart == "Excess Returns & Risk-free Rate Analysis":
            with c22:
                window_size = st.number_input("window-size", min_value = 30, max_value = 90, step = 10)
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 7))
            # Risk-free rate over time
            axes[0, 0].plot(train_df['date_id'], train_df['risk_free_rate'] * 100, linewidth=1.2)
            axes[0, 0].set_title('Risk-Free Rate Over Time', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('date_id')
            axes[0, 0].set_ylabel('Risk-Free Rate (%)')
            axes[0, 0].grid(alpha=0.3)

            # Excess returns over time
            axes[0, 1].plot(train_df['date_id'], train_df['market_forward_excess_returns'], linewidth=0.8, alpha=0.7)
            axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=1)
            axes[0, 1].set_title('Market Forward Excess Returns Over Time', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('date_id')
            axes[0, 1].set_ylabel('Excess Returns')
            axes[0, 1].grid(alpha=0.3)

            # Distribution comparison
            axes[1, 0].hist(train_df['forward_returns'].dropna(), bins=50, alpha=0.5, label='Forward Returns', color='blue')
            axes[1, 0].hist(train_df['market_forward_excess_returns'].dropna(), bins=50, alpha=0.5, 
                            label='Excess Returns', color='orange')
            axes[1, 0].set_title('Returns Distribution Comparison', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Returns')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha = 0.3)

            # Rolling Sharpe ratio
            rolling_sharpe = (train_df['market_forward_excess_returns'].rolling(window = window_size).mean() / 
                            train_df['market_forward_excess_returns'].rolling(window = window_size).std() * np.sqrt(252))
            axes[1, 1].plot(train_df['date_id'], rolling_sharpe, linewidth=1.2)
            axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=1)
            axes[1, 1].set_title(f'{window_size}-Days Rolling Sharpe Ratio', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('date_id')
            axes[1, 1].set_ylabel('Sharpe Ratio')
            axes[1, 1].grid(alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
        
        else:
            fig, ax = plt.subplots(nrows = 1, ncols = 2, 
                                   gridspec_kw={"width_ratios": [2, 1]}, figsize = (16, 5))
            ax[0].plot(calculated_excess)
            ax[0].grid(alpha=0.3)
            ax[0].set_title("Difference (forward_return - risk_free_rate)")
            sns.histplot( calculated_excess, ax = ax[1])
            ax[1].grid(alpha=0.3)
            ax[1].set_title("Histogram of market-forward-excess-return")
            plt.tight_layout()
            st.pyplot(fig)

def volatity_report(train_df):
    st.write("#### 4. Volatility Analysis")
    c1, _, c2 = st.columns([1, 0.1, 5])

    with c1:
        window = st.number_input("`window-size`", min_value=10, max_value=100)
    with c2:
        target_col = st.selectbox(
            'Select target-column',
            ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
        )

    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    rolling_vol = train_df[target_col].rolling(window=window).std()

    ax.plot(train_df['date_id'], rolling_vol, linewidth=1)
    ax.set_title(f'{window}-Period Rolling Volatility of {target_col}')
    ax.set_xlabel('Date ID')
    ax.set_ylabel('Volatility')
    ax.grid(True, alpha=0.3)

    # ===== stats =====
    mean_val = rolling_vol.mean()
    std_val  = rolling_vol.std()
    max_val  = rolling_vol.max()

    stats_text = (
        f"Mean: {mean_val:.6f}\n"
        f"Std:  {std_val:.6f}\n"
        f"Max:  {max_val:.6f}"
    )

    # ===== draw textbox on chart =====
    ax.text(0.01, 0.95, stats_text,
            transform = ax.transAxes, fontsize = 10, verticalalignment = 'top',
            bbox = dict(boxstyle = "round", facecolor = "white", alpha = 0.8)
            )

    st.pyplot(fig)
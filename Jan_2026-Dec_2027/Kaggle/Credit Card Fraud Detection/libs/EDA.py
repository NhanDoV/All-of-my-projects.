import pandas as pd
import seaborn as sns
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# GLOBAL VRS & FUNCs
is_fraud_color_code = {1: "#f27463", 0: "#4c79c2"}
grid_color = "#938FBCCF"
bg_color = "#60A5FA"
axes_color = "#A6BAD4"
merchaint_palette = {
        "Electronics": "#1885A9EF",
        "Clothing": "#31C1D7",
        "Food": "#5E5EBC",
        "Grocery": "#1D5AAF",
        "Travel": "#191279"
    }

def show_plotly_template(fig):
    fig.update_layout(
        plot_bgcolor = "rgba(255,255,255,0.15)",
        paper_bgcolor = "rgba(255,255,255,0)",
        font = dict(color = "#f0f2f6"),
        margin = dict(
                        l = 10, # left
                        r = 10, # right
                        t = 40, # top
                        b = 10 # bottom
                    )
    )
    
    return fig

def styled_df(df):

    num_cols = df.select_dtypes(include="number").columns
    styler = df.style.set_properties(**{
                    "background-color": bg_color,  
                    "color": "#1D287A",           
                    "font-weight": "500",
                    "border": "1px solid #BBDEFB", 
                    "padding": "6px",
                }).set_table_styles([
                    {"selector": "th", "props": [
                        ("background-color", "#09052f"),
                        ("color", "white"),
                        ("font-weight", "bold"),
                        ("text-align", "center"),
                    ]}
                ])

    # center numeric columns
    styler = styler.set_properties(
        subset=num_cols,
        **{"text-align": "center"}
    )

    return styler

# ===================
# BASIC EDA
# ===================
def target_distribution(df, targ_col = 'is_fraud'):
    df_count = df.groupby(targ_col).size().reset_index()
    df_count.columns = [targ_col, 'count']
    
    fig, ax = plt.subplots(nrows = 2, ncols = 1, 
                           figsize = (5, 7.5), facecolor = bg_color)
    # 1. Barplot
    sns.barplot(df_count, x = targ_col, y = 'count', 
                hue = targ_col, palette = is_fraud_color_code,
                ax = ax[0])
    
    for p in ax[0].patches:
        if p.get_height() > 0:
            ax[0].annotate(f'{p.get_height():.0f}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'bottom', 
                            fontsize = 14, color = '#151269'
                        )
    ax[0].set_facecolor(axes_color)
    ax[0].tick_params(axis='both', labelsize = 9) 
    ax[0].set_xlabel('Fraud Status', fontsize = 9)
    ax[0].set_ylabel('Count', fontsize = 9)
    ax[0].set_title("Fraud-distribution", color = "#1D287A", fontweight = "bold")
    ax[0].set_ylim(0, 11000)
    ax[0].grid(color = grid_color)
    
    # 2. Pie-chart of merchaindise
    df_merchant = df.groupby('merchant_category').size()
    wedges, texts, autotexts = ax[1].pie( df_merchant, 
                                          labels = df_merchant.index, 
                                          colors = [merchaint_palette[c] for c in df_merchant.index],
                                          pctdistance = 0.6, autopct = '%1.1f%%', startangle = 90,
                                          wedgeprops = {
                                              "linewidth": 1.5,
                                              "width": 0.8, 
                                              "edgecolor": "white"
                                              }
                                        )
    ax[1].tick_params(labelsize = 11)

    # Edit label category
    for t in texts:
        t.set_fontsize(9)
        t.set_color("black")

    # Edit % inside pie
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(12)
        at.set_fontweight("bold")

    ax[1].set_title("Merchant-category distribution", 
                    color = "#1D287A", fontweight = "bold")

    # Show
    plt.tight_layout()

    st.pyplot(fig, width = 500)

def overall_metric(df):
    
    st.write("### Overall metrics")

    def metric_card(label, value):
        return f"""
        <div class="metric-card metric-center">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """
    data_dims = metric_card("Data dimension <br> rows x columns", 
                            str(df.shape))
    n_missing_rows = metric_card("n_missing_rows", 
                                 df.isnull().sum().max())
    n_duplicated = metric_card("n_duplicated_rows", 
                               df.duplicated().sum() )
    total_amount_isfraud = metric_card("Total amount <br> is_fraud",
                                        df.loc[df['is_fraud'] == 1, 'amount'].sum()
                                    )
    avg_age_isfraud = metric_card("AVG(Age) <br> is_fraud",
                                  df.loc[df['is_fraud'] == 1, 'cardholder_age'].mean().round(2)
                                  )
    avg_score_isfraud = metric_card("AVG(score) <br> is_fraud",
                                    df.loc[df['is_fraud'] == 1, 'device_trust_score'].mean().round(2)
                                    )
    fraud_rate = metric_card("Fraud rate",
                             f"{round(len(df[df['is_fraud'] == 1]) / len(df) * 100, 3)} %"
                            )
    st.markdown(f"""
        <div class="metric-table-grid">
            {data_dims}
            {n_missing_rows}
            {n_duplicated}
            {total_amount_isfraud}
            {avg_age_isfraud}
            {avg_score_isfraud}
            {fraud_rate}
        </div>
    """, unsafe_allow_html=True)

def overview_db(df):
    
    with st.expander("**Review dataset**", expanded = True):
        st.table( styled_df(df.head().set_index("transaction_id")))
    
    df_schema = df.dtypes
    cols_by_dtype = {}

    for col, dtype in df_schema.items():
        cols_by_dtype.setdefault(str(dtype), []).append(col)

    df_schema = pd.DataFrame({
        'dtype': [key for key, _ in cols_by_dtype.items()],
        'count': [len(val) for _, val in cols_by_dtype.items()],
        'cols' : [', '.join(col) for key, col in cols_by_dtype.items()]
    })
    
    with st.expander("**Data schema**", expanded = True):
        st.table( styled_df(df_schema.set_index('dtype')) )

def get_all_distribution(df, num_cols):
    
    with st.expander("**Numerical Univariate Feature Distribution**", 
                     expanded = True):
        cols = st.columns(len(num_cols))
        for idx, col in enumerate(num_cols):
            # -------------------------
            # Histogram
            # -------------------------
            fig1, ax1 = plt.subplots(figsize = (4, 3), facecolor = bg_color)
            sns.histplot(df[col], kde = True, color = "#1728BC", ax = ax1)
            ax1.grid(color = grid_color)
            ax1.set_facecolor(axes_color)

            # display
            cols[idx].pyplot(fig1, width = 'content')
            
            # -------------------------
            # Boxplot
            # -------------------------
            fig2 = px.box(df, x = col, 
                        color = 'is_fraud', color_discrete_map = is_fraud_color_code,
                        height = 300
                        )
            cols[idx].plotly_chart(show_plotly_template(fig2), 
                                width = 'content')
       
def multivariate_report(df, num_cols):
    df['is_fraud'] = df['is_fraud'].astype(str)
    with st.expander("**Multi-variates analytic (2D - 3D scatterplot)**", expanded = True):
        c1, c2, c3, c4 = st.columns([2, 3, 4, 4])
        with c1:
            sel_dim = st.selectbox("Chart_dim", ["2D", "3D"], help = "Chart-dimension")
        with c2:
            x_col = st.selectbox("x_col", num_cols)
        with c3:
            ycol_valid = [col for col in num_cols if col != x_col]
            y_col = st.selectbox("y_col", ycol_valid)
        if sel_dim == "3D":
            zcol_valid = [col for col in num_cols if col not in [x_col, y_col]]
            with c4:
                z_col = st.selectbox("z_col", zcol_valid)
            fig = px.scatter_3d(df, color = 'is_fraud',
                                x = x_col, y = y_col, z = z_col, symbol = 'is_fraud',
                                color_discrete_map = {"1": "#f27463", "0": "#4c79c2"})
        else:
            fig = px.scatter(df, x = x_col, y = y_col, 
                                color = 'is_fraud', symbol = 'is_fraud',
                                color_discrete_map = {"1": "#f27463", "0": "#4c79c2"})
        fig = show_plotly_template(fig)
        st.plotly_chart(fig)

def binary_feature_report(df):

    with st.expander("**Binary Feature analytic**", expanded = True):
        temp = df.groupby(['is_fraud', 'foreign_transaction', 'location_mismatch']).size().reset_index()
        temp.columns = ['is_fraud', 'foreign_transaction', 'location_mismatch', 'count']
        c1, _, c2 = st.columns([1, 0.02, 1])
        with c1:
            st.write("""
                     Denoted that :
                     - `F` : fraud
                     - `T` : foreign transaction
                     - `L` : location mismatch
                    """)
            st.write("###### Detailed table")
            st.table( styled_df(temp.set_index(['is_fraud', 'foreign_transaction', 'location_mismatch']) ) )
        with c2:
            st.write('**Basic statistics**')
            N_f = df.groupby('is_fraud').size().loc["1"]
            pf = (N_f / len(df)) * 100
            pt = (df.groupby('foreign_transaction').size().loc[1] / len(df)) * 100
            pl = (df.groupby('location_mismatch').size().loc[1] / len(df)) * 100
            st.latex(rf"""
                     \color{{DarkBlue}} 
                     \mathbb{{P}} \left( F \right) = {pf:.2f} \% \qquad 
                     \mathbb{{P}} \left( T \right) = {pt:.2f} \% \qquad 
                     \mathbb{{P}} \left( L \right) = {pl:.2f} \%
                    """)
            
            st.write('**Conditional statistics**')
            N_ftl = temp.loc[ (temp['is_fraud'] == "1") & (temp['foreign_transaction'] == 1) & (temp['location_mismatch'] == 1), 'count'].sum()
            N_ft = temp.loc[ (temp['is_fraud'] == "1") & (temp['foreign_transaction'] == 1), 'count'].sum()
            N_t = temp.loc[ temp['foreign_transaction'] == 1, 'count' ].sum()
            N_fl = temp.loc[ (temp['is_fraud'] == "1") & (temp['location_mismatch'] == 1), 'count'].sum() 
            N_l = temp.loc[ temp['location_mismatch'] == 1, 'count' ].sum()
            N_tl = temp.loc[ (temp['foreign_transaction'] == 1) & (temp['location_mismatch'] == 1), 'count'].sum() 
            
            st.latex(rf"""
                     \color{{DarkBlue}} 
                     \mathbb{{P}} \left( F | T \right) = \dfrac{{ {N_ft} }} {{ {N_t} }} = {100* N_ft / N_t:.3f} \% \qquad 
                     \mathbb{{P}} \left( F | L \right) = \dfrac{{ {N_fl} }} {{ {N_l} }} = {100 * N_fl / N_l:.3f} \%
                    """)

            st.latex(rf"""
                     \color{{DarkBlue}}
                     \mathbb{{P}} \left( T | F \right) = \dfrac{{ {N_ft} }} {{ {N_f} }} = {100* N_ft / N_f:.2f} \% \qquad 
                     \mathbb{{P}} \left( L | F \right) = \dfrac{{ {N_fl} }} {{ {N_f} }} = {100 * N_fl / N_f:.2f} \%
                    """)
            
            st.latex(rf"""
                    \color{{Blue}}
                    \mathbb{{P}} \left( F | T, L \right) = \tfrac{{ {N_ftl} }} {{ {N_tl} }} = {100* N_ftl / N_tl:.2f} \% \quad
                    \mathbb{{P}} \left( T, L | F \right) = \tfrac{{ {N_ftl} }} {{ {N_f} }} = {100* N_ftl / N_f:.2f} \% 
                    """)

            st.write("**Lift**")
            st.latex(rf"""
                    \color{{DarkBlue}}
                     \begin{{array}}{{ccl}}
                        \text{{Lift}}(T) & := & \dfrac{{ \mathbb{{P}} \left( F | T \right) }} {{ \mathbb{{P}} (F) }} = {100 * (N_ft / N_t) / pf :.4f} \% \\ \\
                        \text{{Lift}}(L) & := & \dfrac{{ \mathbb{{P}} \left( F | L \right) }} {{ \mathbb{{P}} (F) }} = {100 * (N_fl / N_l) / pf :.4f} \%
                     \end{{array}}
                    """)
            
def merchant_report(df):
    with st.expander("**Merchandise report**", expanded = True):
        c1, _, c2, _, c3 = st.columns([3, 0.1, 2, 0.1, 2])
        with c1:
            st.write("**Frequency table**")
            df_merchant = pd.pivot_table(df, 
                                        index = 'is_fraud', columns = 'merchant_category',
                                        aggfunc = 'count', values = 'transaction_id',
                                        margins = True, margins_name = 'Total'
                                    )
            st.table(styled_df(df_merchant))
    
        cates = df_merchant.columns[:-1]
        for cate in cates:
            p_AB = len( df[ (df['merchant_category'] == cate) & (df['is_fraud'] == "1") ] )
            p_B = len(df[df['is_fraud'] == "1"])
            p_A = len(df[df['merchant_category'] == cate])

            with c2:
                st.latex(rf"""
                        \color{{DarkBlue}}
                        \mathbb{{P}} ( \text{{Fraud}} | \text{{ {cate} }} ) = {p_AB} / {p_B} = {(p_AB / p_B * 100):.2f} \%
                        """)

            with c3:
                st.latex(rf"""
                        \color{{Blue}}
                        \mathbb{{P}} ( \text{{ {cate} }} | \text{{Fraud}} ) = {p_AB} / {p_A} = {(p_AB / p_A * 100):.2f}
                        """)
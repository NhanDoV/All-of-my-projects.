import streamlit as st
import pandas as pd

st.set_page_config(page_title="Clean Employee Null Values", page_icon="💣", layout="wide")
descr_col, hint_col = st.columns([4, 3], gap='medium')

with descr_col:
    with st.expander("**:violet[DESCRIPTION]**", expanded=True):
        st.markdown(
            """
            E-commerce marketing campaigns tier customers based on cumulative spend thresholds: 
            - total_spent >= 1000.0  ⇒  `"Platinum"` 
            - total_spent >= 500.0  ⇒ `"Gold"` 
            - total_spent >= 200.0  ⇒ `"Silver"` 
            - otherwise  ⇒ `"Bronze"`
            
            Add a `tier` column, and sort the final DataFrame by `total_spent` DESC, then `customer_id` ASC.

            <span style="color: #89CFF0"> **Input dataframe:** </span> `customers_df` or `customers` table
            """,
            unsafe_allow_html=True
        )
with hint_col:
    with st.expander("**:violet[INPUT SCHEMA]**", expanded=True):
        st.dataframe(pd.DataFrame({
            'customer_id': ['INT'],
            'customer_name': ['STRING'],
            'total_spent': ['INT'],
        }), hide_index=True)

    with st.expander("**:violet[EXPECTED OUTPUT SCHEMA]**", expanded=True):
        st.dataframe(pd.DataFrame({
            'customer_id': ['INT'],
            'customer_name': ['STRING'],
            'total_spent': ['INT'],
            'tier': ['STRING']
        }), hide_index=True)

# ---------------------
pyspark_cd = """
    customers_df.withColumn(
                   "tier", 
                   F.when(F.col('total_spent') >= 1000, 'Platinum')\\
                    .when(F.col('total_spent') >= 500, 'Gold') \\
                    .when(F.col('total_spent') >= 200, 'Silver') \\
                    .otherwise('Bronze')
                ).orderBy(
                    F.col('total_spent').desc(),
                    F.col('customer_id').asc()
                )
"""
sql_code = """
    SELECT *,
           CASE
                WHEN total_spent >= 1000 THEN 'Platinum'
                WHEN total_spent >= 1000 THEN 'Gold'
                WHEN total_spent >= 1000 THEN 'Silver'
                ELSE 'Bronze'
    FROM customers
    ORDER BY total_spent DESC, customer_id ASC;
"""

st.html(f"""
    <style>
    .playground-title {{
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 20px;
        font-family: Consolas, monospace;
    }}

    .code-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 40px;
        align-items: stretch;
    }}

    .code-card {{
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 12px;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        min-width: 0;
    }}

    .code-header {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 12px 12px;
        background: #181825;
        border-bottom: 1px solid #313244;
        font-family: Consolas, monospace;
        font-size: 16px;
        font-weight: 700;
        color: #cdd6f4;
    }}

    .code-lang {{
        margin-left: auto;
        color: #9399b2;
        font-size: 14px;
        font-weight: 400;
    }}

    .python-label {{
        color: #89b4fa;
    }}

    .sql-label {{
        color: #a6e3a1;
    }}

    .code-body {{
        flex: 1;
        display: flex;
        align-items: center;
        font-size: 14px;
        padding: 15px;
        overflow-x: auto;
    }}

    .code-body pre {{
        margin: 0;
        white-space: pre;
        font-family: Consolas, monospace;
        font-size: 14px;
        padding-left: 50px;
        line-height: 1.8;
        color: #cdd6f4;
    }}

    .code-card {{
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 12px;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        min-width: 0;

        transition:
            transform 0.25s ease,
            border-color 0.25s ease,
            box-shadow 0.25s ease;
    }}

    .code-card:hover {{
        transform: translateY(-5px);
        border-color: #585b70;
        box-shadow:
            0 10px 25px rgba(0, 0, 0, 0.35),
            0 0 12px rgba(137, 180, 250, 0.08);
    }}

    @media (max-width: 640px) {{
        .code-grid {{
            grid-template-columns: 1fr;
        }}
    }}
    </style>

    <div class="playground-title">Playground</div>

    <div class="code-grid">
        <div class="code-card">
            <div class="code-header">
                <span class="python-label">●</span>
                <span>PySpark</span>
                <span class="code-lang">Python</span>
            </div>

            <div class="code-body">
                <pre> {pyspark_cd} </pre>
            </div>
        </div>

        <div class="code-card">
            <div class="code-header">
                <span class="sql-label">●</span>
                <span>SQL</span>
                <span class="code-lang">SQL</span>
            </div>

            <div class="code-body">
                <pre> {sql_code} </pre>
            </div>
        </div>
    </div>
""")

st.divider()

# ---------------------
test_col, inp_col, otp_col = st.columns([1.25, 3.75, 4], gap='large')

# tạo dictionary lưu input vs output
import numpy as np

data_dict = {
    'Test case 1': pd.DataFrame({
                        "customer_id": [1, 2, 5, 19, 29],
                        "customer_name" : ['Alice', 'Bob', 'Clare', 'David', 'Elena'],
                        "total_spent": [1250, 288, 121, 80, 611],
                    }),
    'Test case 2': pd.DataFrame({
                        "customer_id": [29, 121, 244, 288, 333, 411],
                        "customer_name" : ['Diana', 'Evan', 'Nancy', 'Noah', 'Tim', 'Xavier'],
                        "total_spent": [1200, 980, 450, 210, 80, 600],
                    })
}

with test_col:
    st.subheader("Results")
    input_df_case = st.selectbox("Chọn một trong các test case sau", [f"Test case {i}" for i in range(1, 3)])

with inp_col:
    with st.expander("**:violet[Input]**", expanded=True):
        inp_df = data_dict[input_df_case]
        _, ic, _ = st.columns([1,19,1])
        with ic:
            st.dataframe(inp_df, hide_index=True)

with otp_col:
    with st.expander("**:violet[Output]**", expanded=True):
        _, oc, _ = st.columns([1,19,1])
        with oc:
            inp_df['tier'] = inp_df['total_spent'].case_when(
                [
                    (inp_df.eval('total_spent >= 1000.0'), 'Platinum' ),
                    (inp_df.eval('total_spent >= 500.0'), 'Gold' ),
                    (inp_df.eval('total_spent >= 200.0'), 'Silver' ),
                    (inp_df.eval('total_spent < 200.0'), 'Bronze' )
                ]
            )
            st.dataframe(inp_df, hide_index=True)
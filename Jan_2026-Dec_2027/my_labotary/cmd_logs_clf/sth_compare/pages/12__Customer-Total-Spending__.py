import streamlit as st
import pandas as pd

st.set_page_config(page_title="Customer TotalSpending", page_icon="💣", layout="wide")
descr_col, hint_col = st.columns([3, 4], gap='medium')

with descr_col:
    with st.expander("**:violet[DESCRIPTION]**", expanded=True):
        st.markdown(
            """
            E-commerce CRM systems must compute total order expenditure for every registered customer.
            - Join the `customers_df` with `orders_df` and calculate `total_spent` per customer.
            - Customers who have placed no orders must still appear in the output with `total_spent = 0.0`.

            <span style="color: #89CFF0"> **Input dataframe:** </span> 
            - `customers_df` or `customers` table
            - `orders_df` or `orders` table
            """,
            unsafe_allow_html=True
        )

with hint_col:
    with st.expander("**:violet[INPUT SCHEMA]**", expanded=True):
        emp_df, dept_df = st.columns([2, 3], gap='large')
        with emp_df:
            st.write("**:orange[customers_df]**")
            st.dataframe(pd.DataFrame({
                'customer_id': ['INT'],
                'customer_name': ['STRING']
            }), hide_index=True)
        with dept_df:
            st.write("**:orange[orders_df]**")
            st.dataframe(pd.DataFrame({
                'order_id ': ['INT'],
                'customer_id ': ['INT'],
                'amount': ['DOUBLE'],
            }), hide_index=True)

    with st.expander("**:violet[EXPECTED OUTPUT SCHEMA]**", expanded=True):
        st.dataframe(pd.DataFrame({
            'customer_id': ['INT'],
            'customer_name': ['STRING'],
            'total_spent': ['DOUBLE'],
        }), hide_index=True)

# ---------------------
pyspark_cd = """
    customers_df.join(
            orders_df, on = 'customer_id', how = 'left'
        ).fillna(0) \\
        .groupby('customer_id', 'customer_name') \\
        .agg(
            F.sum('amount').alias('total_spent')
        )
"""
sql_code = """
    SELECT customer_id, customer_name,
            SUM( COALESCE(amount, 0) ) AS total_spent
    FROM (
            SELECT c.customer_id, c.customer_name, o.amount
            FROM customers c
            LEFT JOIN orders o
                ON c.customer_id - o.customer_id
        )
    GROUP BY customer_id, customer_name
    ORDER BY customer_id;
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
inp_col, otp_col = st.columns([5, 3], gap='large')

# tạo dictionary lưu input vs output
import numpy as np

with inp_col:
    with st.expander("**:violet[Input]**", expanded=True):
        customers_df = pd.DataFrame({
                    'customer_id': [1,2,3,4],
                    'customer_name': ['Alice', 'Bob', 'Carol', 'David']
              })
        orders_df = pd.DataFrame({
                    'order_id': [101, 102, 103, 104, 105, 106],
                    'customer_id': [1, 1, 3, 4, 4, 1],
                    'amount': [150, 250, 80, 140, 50, 100]
            })
        ecol, dcol = st.columns([2, 3], gap='large')
        with ecol:
            st.write("customers_df")
            st.dataframe(customers_df, hide_index=True)
        with dcol:
            st.write("orders_df")
            st.dataframe(orders_df, hide_index=True)

with otp_col:
    with st.expander("**:violet[Output]**", expanded=True):
        _, oc, _ = st.columns([1,19,1])
        with oc:
            res = customers_df.merge(
                        orders_df, on='customer_id', how='left'
                    ).fillna(0)\
                    .groupby(['customer_id', 'customer_name'])\
                    .agg(total_spent = ('amount', 'sum')).reset_index()
            st.dataframe(res, hide_index=True)
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Clean Employee Null Values", page_icon="💣", layout="wide")
descr_col, hint_col = st.columns([4, 3], gap='medium')

with descr_col:
    with st.expander("**:violet[DESCRIPTION]**", expanded=True):
        st.markdown(
            """
            Filter the employee dataset to return records meeting both of the following conditions: 
            - 1. `department == "IT"`
            - 2. `salary >= 70000.0`
            
            <span style="color: orange"> **Important PySpark Rule:** </span>
            - In PySpark Column expressions, always use bitwise boolean operators (&, |, ~) wrapped in parentheses (... & ...). 
            - Do not use Python's and / or.

            <span style="color: #89CFF0"> **Input dataframe:** </span> `customers_df` or `customers` table
            """,
            unsafe_allow_html=True
        )
with hint_col:
    with st.expander("**:violet[INPUT SCHEMA]**", expanded=True):
        st.dataframe(pd.DataFrame({
            'employee_id': ['INT'],
            'name': ['STRING'],
            'department': ['STRING'],
            'salary': ['DOUBLE'],
        }), hide_index=True)

    with st.expander("**:violet[EXPECTED OUTPUT SCHEMA]**", expanded=True):
        st.dataframe(pd.DataFrame({
            'employee_id': ['INT'],
            'name': ['STRING'],
            'department': ['STRING'],
            'salary': ['DOUBLE'],
        }), hide_index=True)

# ---------------------
pyspark_cd = """
    employees_df.filter(
            (F.col('department') == 'IT) & 
            (F.col('salary') >= 70_000.0)
        )
"""
sql_code = """
    SELECT * FROM employees
    WHERE (department = 'IT) AND
          (salary >= 70000.0)
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
                        'employee_id': [1,2,3,4],
                        'name': ['Alice', 'Bob', 'Charlie', 'David'],
                        'department': ['IT', 'IT', 'Finance', 'IT'],
                        'salary': [85_000, 62_000, 90_000, 70_000]
                    }),
    'Test case 2': pd.DataFrame({
                        'employee_id': [10,12,23],
                        'name': ['Elly', 'Eve', 'Frank'],
                        'department': ['IT', 'HR', 'IT'],
                        'salary': [90_000, 69_999, 70_001]
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
            inp_df = inp_df[
                            (inp_df['department'] == 'IT') & 
                            (inp_df['salary'] > 70_000.0)
                        ]
            st.dataframe(inp_df, hide_index=True)
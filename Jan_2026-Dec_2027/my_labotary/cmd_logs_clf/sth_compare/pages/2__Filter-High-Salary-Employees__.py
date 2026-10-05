import streamlit as st
import pandas as pd

st.set_page_config(page_title="Select Employee Columns", page_icon="💣", layout="wide")
descr_col, hint_col = st.columns([4, 3], gap='medium')

with descr_col:
    with st.expander("DESCRIPTION", expanded=True):
        st.markdown(
            """
                - Compensation analytics teams need to identify compensation outliers across the company. 
                - Return all employee records whose salary is strictly greater than 50000.0.

                <span style="color: #89CFF0"> **Input dataframe:** </span> `employees_df` or `employees` table

                <span style="color: #088F8F"> **Input schema:** </span>
            """,
            unsafe_allow_html=True
        )
        _, schema_tab, _ = st.columns([1,9,1])
        with schema_tab:
            st.dataframe(pd.DataFrame({
                'employee_id': ['INT'],
                'name': ['STRING'],
                'department': ['STRING'],
                'salary': ['DOUBLE']
            }), hide_index=True)

with hint_col:
    with st.expander("Learning Objectives", expanded=True):
        st.markdown(
            """
                1. Apply relational filtering with df.filter() or `df.where()` / `df.filter`. 
                2. Understand predicate pushdown where filters are evaluated as close to disk as possible.
            """
            , unsafe_allow_html=True
        )

# ---------------------
pyspark_cd = """
    employees_df.filter( F.col('salary') > 50000.0 )
"""
sql_code = """
    SELECT name, salary
    FROM employees
    WHERE salary > 50000.0;
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
test_col, inp_col, otp_col = st.columns([1.75, 4, 4], gap='large')

# tạo dictionary lưu input vs output
data_dict = {
    'Test case 1': pd.DataFrame({
                        "employee_id": [1, 2, 3],
                        "name" : ['Alice', 'Bob', 'Charlie'],
                        "department": ['Enginering', 'HR', 'Finance'],
                        "salary": [95_000, 62_000, 88_000],
                    }),
    'Test case 2': pd.DataFrame({
                        "employee_id": [10, 22],
                        "name" : ['Diana', 'Helen'],
                        "department": ['Legal', 'Manager'],
                        "salary": [120_000, 116_000],
                    }),
    'Test case 3': pd.DataFrame({
                        "employee_id": [101, 102, 103],
                        "name" : ['Elena', 'Frank', 'Grace'],
                        "department": ['Data', 'Design', 'Marketing'],
                        "salary": [105_000, 78_000, 84_000],
                    })
}

with test_col:
    st.subheader("Results")
    input_df_case = st.selectbox("Chọn một trong các test case sau", [f"Test case {i}" for i in range(1, 4)])

with inp_col:
    with st.expander("**:violet[Input]**", expanded=True):
        inp_df = data_dict[input_df_case]
        _, ic, _ = st.columns([1,9,1])
        with ic:
            st.dataframe(inp_df, hide_index=True)

with otp_col:
    with st.expander("**:violet[Output]**", expanded=True):
        _, oc, _ = st.columns([1,9,1])
        with oc:
            st.dataframe(inp_df[inp_df['salary'] > 50_000.0], hide_index=True)
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Clean Employee Null Values", page_icon="💣", layout="wide")
descr_col, hint_col = st.columns([3, 4], gap='medium')

with descr_col:
    with st.expander("**:violet[DESCRIPTION]**", expanded=True):
        st.markdown(
            """
            - In normalized relational databases, organizational entities are split into separate dimension and fact tables. 
            - Perform an inner join between the employees dataset and departments dataset on `dept_id`.

            <span style="color: #89CFF0"> **Input dataframe:** </span> 
            - `employees_df` or `employees` table
            - `departments_df` or `departments` table
            """,
            unsafe_allow_html=True
        )

with hint_col:
    with st.expander("**:violet[INPUT SCHEMA]**", expanded=True):
        emp_df, dept_df = st.columns([3, 2], gap='large')
        with emp_df:
            st.write("**:orange[employees_df]**")
            st.dataframe(pd.DataFrame({
                'employee_id': ['INT'],
                'name': ['STRING'],
                'dept_id ': ['INT'],
                'salary': ['DOUBLE'],
            }), hide_index=True)
        with dept_df:
            st.write("**:orange[departments_df]**")
            st.dataframe(pd.DataFrame({
                'dept_id ': ['INT'],
                'department_name': ['STRING'],
            }), hide_index=True)

    with st.expander("**:violet[EXPECTED OUTPUT SCHEMA]**", expanded=True):
        st.dataframe(pd.DataFrame({
            'employee_id': ['INT'],
            'name': ['STRING'],
            'department_name': ['STRING'],
            'salary': ['DOUBLE'],
        }), hide_index=True)

# ---------------------
pyspark_cd = """
    employees_df.join(
            departments_df,
            how = 'inner',
            on = 'dept_id'
        )
"""
sql_code = """
    SELECT 
        e.employee_id, e.name,
        d.department_name, 
        e.salary
    FROM employees e
    INNER JOIN departments d
        ON e.dept_id = d.dept_id;
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
        employees_df = pd.DataFrame({
                'employee_id': [1, 2, 3, 4, 5, 6],
                'name': ['Alice', 'Bob', 'Carol', 'Charile', 'David', 'Elly'],
                'dept_id': [10, 20, 10, 99, 10, 20],
                'salary': [90_000, 72_000, 105_000, 60_000, 121_000, 75_000]            
              })
        departments_df = pd.DataFrame({
                'dept_id': [10, 20, 30, 80, 90],
                'department_name': ['Engineering', 'Marketing', 'HR', 'Finance', 'Design']
            })
        ecol, dcol = st.columns([2, 1], gap='large')
        with ecol:
            st.write("employees_df")
            st.dataframe(employees_df, hide_index=True)
        with dcol:
            st.write("departments_df")
            st.dataframe(departments_df, hide_index=True)

with otp_col:
    with st.expander("**:violet[Output]**", expanded=True):
        _, oc, _ = st.columns([1,19,1])
        with oc:
            selected_cols = ['employee_id', 'name', 'department_name', 'salary']
            res = employees_df.merge(
                                    departments_df, 
                                    on='dept_id', how='inner'
                                )[selected_cols]
            st.dataframe(res, hide_index=True)
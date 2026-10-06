import streamlit as st
import pandas as pd

st.set_page_config(page_title="Calculate Employee Bonus", page_icon="💣", layout="wide")
descr_col, hint_col = st.columns([4, 3], gap='medium')

with descr_col:
    with st.expander("**:violet[DESCRIPTION]**", expanded=True):
        st.markdown(
            """
            - During annual compensation reviews, each employee receives a performance bonus equal to `10% (0.10)` of their **base salary**.
            - Derive a new column `bonus` equal to `round(salary * 0.10, 2)`.

            <span style="color: #89CFF0"> **Input dataframe:** </span> `employees_df` or `employees` table
            """,
            unsafe_allow_html=True
        )
with hint_col:
    with st.expander("**:violet[INPUT SCHEMA]**", expanded=True):
        st.dataframe(pd.DataFrame({
            'employee_id': ['INT'],
            'name': ['STRING'],
            'salary': ['DOUBLE']
        }), hide_index=True)

    with st.expander("**:violet[EXPECTED OUTPUT SCHEMA]**", expanded=True):
        st.dataframe(pd.DataFrame({
            'employee_id': ['INT'],
            'name': ['STRING'],
            'salary': ['DOUBLE'],
            'bonus': ['DOUBLE']
        }), hide_index=True)

# ---------------------
pyspark_cd = """
    employees_df.withColumn(
                            "bonus", 
                            F.round(F.col("salary") * 0.10, 2)
                        )
"""
sql_code = """
    SELECT *,
           ROUND( salary * 0.10, 2 ) AS bonus 
    FROM employees;
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
                        "employee_id": [1, 2, 3, 4, 5],
                        "name" : ['Alice', 'Bob', 'Charlie', 'Elena', 'Emma'],
                        "salary": [25_450, 16_432.1, 8_500, 9_526.2, 6_200],
                    }),
    'Test case 2': pd.DataFrame({
                        "employee_id": [10, 22, 29, 39],
                        "name" : ['Diana', 'Helen', 'Nancy', 'Pedro'],
                        "salary": [121_244, 41_600, 93_500, 26_525],
                    }),
    'Test case 3': pd.DataFrame({
                        "employee_id": [101, 102, 103],
                        "name" : ['Elena', 'Frank', 'Grace'],
                        "salary": [105_121.2, 38_244.1, 54_123],
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
            inp_df['bonus'] = round(0.10 * inp_df['salary'], 2)
            st.dataframe(inp_df, hide_index=True)
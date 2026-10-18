import streamlit as st
import pandas as pd

st.set_page_config(page_title="Customer TotalSpending", page_icon="💣", layout="wide")
descr_col, hint_col = st.columns([3, 4], gap='medium')

with descr_col:
    with st.expander("**:violet[DESCRIPTION]**", expanded=True):
        st.markdown(
            """
            Academic scorecards store grades in normalized key-value format. 
            - Pivot student subject marks so each subject becomes an individual column.
            - Subjects to pivot: `English`, `History`, `Maths`, `Physics`, `Science`

            Also compute a `TOTAL` column equal to the sum of marks across all 5 subjects

            <span style="color: #89CFF0"> **Input dataframe:** </span> `marks_df` or `marks` table
            """,
            unsafe_allow_html=True
        )

with hint_col:
    with st.expander("**:violet[INPUT SCHEMA]**", expanded=True):
        st.write("**:orange[marks_df]**")
        st.dataframe(pd.DataFrame({
            'student_id ': ['INT'],
            'subject': ['STRING'],
            'marks': ['INT'],
        }), hide_index=True)


    with st.expander("**:violet[EXPECTED OUTPUT SCHEMA]**", expanded=True):
        st.dataframe(pd.DataFrame({
            'student_id': ['INT'],
            'English': ['INT'],
            'History': ['INT'],
            'Maths': ['INT'],
            'Physics': ['INT'],
            'Science': ['INT'],
            'TOTAL': ['INT'],
        }), hide_index=True)

# ---------------------
pyspark_cd = """
    subjects = ['English', 'History', 'Maths', 'Physics', 'Science']
    pivoted = marks_df.groupBy('student_id')\\
                      .pivot('subject', subjects)\\
                      .sum('marks')
    total_expr = sum(
                      F.coalesce(F.col(sub), F.lit(0.0)) 
                                      for sub in subjects
                    )
    pivoted.withColumn('TOTAL', total_expr)
"""
sql_code = """
    SELECT *,
           English + History + Maths + Physics + Science AS TOTAL
    FROM (
            SELECT * FROM marks
            PIVOT (
                    SUM(marks) FOR subject IN ('English', 'History', 
                                               'Maths', 'Physics', 
                                               'Science')
                )
    ) 
    ORDER BY student_id;
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
inp_col, otp_col = st.columns([3, 4], gap='large')

# tạo dictionary lưu input vs output
import numpy as np

with inp_col:
    with st.expander("**:violet[Input]**", expanded=True):
        marks_df = pd.DataFrame({
                'student_id': [1,1,1,1,1, 2,2,2,2,2],
                'subject': ['Maths', 'Physics', 'Science', 'English', 'History', 'Maths', 'Physics', 'Science', 'English', 'History'],
                'marks': [ 95, 88, 92, 85, 90, 70, 75, 80, 65, 70]
              })
        _, ic, _ = st.columns([1,11,1])
        with ic:
            st.dataframe(marks_df, hide_index=True)

with otp_col:
    with st.expander("**:violet[Output]**", expanded=True):
        _, oc, _ = st.columns([1,19,1])
        with oc:
            df = marks_df.copy()
            df = pd.pivot_table(df, index='student_id', columns='subject', values='marks')\
                .astype(int)
            df['TOTAL'] = df.sum(axis=1)
            df = df.reset_index()
            st.dataframe(df, hide_index=True)
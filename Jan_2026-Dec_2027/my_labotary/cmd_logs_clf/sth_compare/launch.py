import streamlit as st

st.set_page_config(
    page_title="SPARK REVIEW",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.hero {
    padding: 65px 40px 55px 40px;
    text-align: center;
    border-radius: 20px;
    margin-bottom: 35px;
}

.hero h1 {
    font-size: 56px;
    font-weight: 800;
    margin: 10px 0 8px 0;
    font-family: Consolas, monospace;
}

.hero .subtitle {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 10px;
}

.hero .description {
    font-size: 17px;
    opacity: 0.7;
    line-height: 1.6;
}

.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    border: 1px solid rgba(128,128,128,.3);
    font-size: 13px;
    margin-bottom: 15px;
    letter-spacing: 0.5px;
}
</style>

<div class="hero">

<div class="badge">PYSPARK • SQL • COMPARISON PLAYGROUND</div>

<h1>✨ SPARK REVIEW</h1>

<div class="subtitle">
PySpark ↔ SQL
</div>

<div class="description">
Same problem. Two ways to solve it.<br>
Compare DataFrame operations with their SQL equivalents.
</div>

</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 01 · Understand
    Start with a real data problem.
    """)

with col2:
    st.markdown("""
    ### 02 · Compare
    See how the same operation is expressed
    in PySpark and SQL.
    """)

    _, c, _ = st.columns([1,4,1])
    with c:
        st.markdown("""
        <div class="comparison-map">
        <table>
        <tr>
            <th>PySpark</th>
            <th></th>
            <th>SQL</th>
        </tr>
        <tr>
            <td><code>select()</code></td>
            <td>↔</td>
            <td><code>SELECT</code></td>
        </tr>
        <tr>
            <td><code>filter() / where()</code></td>
            <td>↔</td>
            <td><code>WHERE</code></td>
        </tr>
        <tr>
            <td><code>groupBy()</code></td>
            <td>↔</td>
            <td><code>GROUP BY</code></td>
        </tr>
        <tr>
            <td><code>join()</code></td>
            <td>↔</td>
            <td><code>JOIN</code></td>
        </tr>
        <tr>
            <td><code>orderBy()</code></td>
            <td>↔</td>
            <td><code>ORDER BY</code></td>
        </tr>
        <tr>
            <td><code>Window</code></td>
            <td>↔</td>
            <td><code>OVER()</code></td>
        </tr>
        </table>

        </div>
        """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    ### 03 · Practice
    Run examples and verify the expected result.
    """)
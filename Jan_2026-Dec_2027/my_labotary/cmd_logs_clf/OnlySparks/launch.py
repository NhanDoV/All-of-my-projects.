import streamlit as st
import pandas as pd

# FILE NÀY NÊN TỔN TẠI Ở 1 FOLDER ĐỘC LẬP HOÀN TOÀN TRÊN REPO

topic_title_dict = {
    'Basic': [
            'Bs01. Top High Salary Employees', 
            'Bs02. Big Economy but small area',
            'Bs03. BroadCast small dataset joinning',
            'Bs04. Salt_key for skewed data',
            'Bs05. Drop duplicates then Repartition'
        ],
    'GroupAgg': [
            'GA01. Monthly Transaction',
            'GA02. Department Salary Summary'
        ],
    'GroupPivot': [
            'GP01. Pivot Student Subject Marks',
            'GP02. Purchase per category for each customer'
        ],
    'Timestamp Processor': [
            'TP01. Subscription Expiry',
            'TP02. Extract daypart'
        ],
    'Regex Clause': [
            'RC01. Invalid descriptions',
            'RC02. Transaction behavior pattern'
        ],
    'Window': [
            'Wd01. Rising Temperatures',
            'Wd02. End-to-end Sales Analytic Pipeline'
    ]
}

# Cột slidebar để chọn topic và title
topic_titles = list(topic_title_dict.keys())

with st.sidebar:
    topic = st.selectbox('Chọn topic', topic_titles)
    title = st.selectbox('Chọn title', topic_title_dict[topic])

# Phần app
st.title('My App')
st.set_page_config(page_title="OnlySpark", page_icon="⚡", layout="wide")

if topic == 'Basic':
    if title == 'Bs01. Top High Salary Employees':

        df = pd.DataFrame({
            'employee_id': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'name': ['Alice', 'Anna', 'Ben', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace'],
            'department': ['HR', 'Digital', 'Design', 'HR', 'Finance', 'Digial', 'HR', 'Finance', 'Design'],
            'hire_date': ['2023-01-23', '2022-07-15', '2021-03-08', '2020-11-02', '2019-06-24', '2022-02-14', '2023-05-09', '2018-09-17', '2021-12-01'],
            'salary': [98000, 62500, 73200, 69500, 121000, 86500, 59000, 118000, 92000]
        })

        c1, c2 = st.columns(2)
        with c1:
            st.write(f'##### {title}')
            st.write('Nội dung bài toán: Tìm 5 nhân viên có mức lương cao nhất trong công ty.')
            st.write('Dữ liệu đầu vào: Bảng `employees_df` với các cột: `employee_id, name, department, hire_date, salary`.')
        with c2:
            st.code(
"""
    from pyspark.sql import functions as F
    
    employees_df.orderBy(F.col('salary').desc()).show(5)
""", language="python")

        st.divider()
        ic, _, oc = st.columns([9,1,9], gap='large')
        with ic:
            st.write("##### Input")
            st.dataframe(df, hide_index=True)
        with oc:
            st.write("##### Output")
            res = df.copy()
            res = df.sort_values(by='salary', ascending=False).head(5)
            st.dataframe(res, hide_index=True)

    if title == "Bs02. Big Economy but small area":
        df = pd.DataFrame({
            "country": ["Netherlands", "Belgium", "Switzerland", "Israel", "Singapore", "Vietnam", "Brazil", "Russia", "Australia", "Japan", "Denmark", "Norway", "Ireland", "Austria", "Czechia", "Portugal", "Greece", "Malaysia", "Thailand", "Indonesia"],
            "continent": ["Europe", "Europe", "Europe", "Asia", "Asia", "Asia", "South America", "Europe", "Oceania", "Asia", "Europe", "Europe", "Europe", "Europe", "Europe", "Europe", "Europe", "Asia", "Asia", "Asia"],
            "GDP_Billion_USD": [1150, 660, 940, 530, 530, 470, 2170, 2180, 1750, 4200, 420, 490, 560, 520, 340, 300, 250, 430, 540, 1370],
            "area_km2": [41543, 30528, 41284, 22072, 728, 331212, 8514877, 17098246, 7692024, 377975, 42933, 385207, 70273, 83879, 78865, 92090, 131957, 330803, 513120, 1904569]
        })
        df = df.sort_values(by='country')

        c1, c2 = st.columns(2)
        with c1:
            st.write(f'##### {title}')
            st.write('''
                        Nội dung bài toán: 
                        - Tìm các quốc gia có diện tích dưới 50000 $km^2$ nhưng có GDP trên 200 tỷ USD.
                        - Sắp xếp theo kinh tế giảm dần, diện tích giảm dần và theo thứ tự tăng dần alphabet tên quốc gia
                    ''')

        with c2:
            st.code(
"""
    world_df.filter(
            (F.col('GDP_Billion_USD') > 200) & (F.col('area_km2') < 50000)
           ).orderBy(
                F.col('GDP_Billion_USD').desc(),
                F.col('area_km2').desc(),
                F.col('country').asc() 
           ).show()
"""                
            )

        ic, _, oc = st.columns([6, 1, 6], gap='large')
        with ic:
            st.dataframe(df, hide_index=True)

        with oc:
            res = df.copy()
            res = res[(res['GDP_Billion_USD'] > 200) & (res['area_km2'] < 50000)]\
                    .sort_values(
                        by=['GDP_Billion_USD', 'area_km2', 'country'],
                        ascending=[False, False, True]
                    )
            st.dataframe(res, hide_index=True)

    if title == "Bs03. BroadCast small dataset joinning":
        st.write(f'##### {title}')
        
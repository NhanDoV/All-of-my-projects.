import pandas as pd
import streamlit as st

# ==================================================================
# DATA OVERVIEW & DESCRIPTION
# ==================================================================
def data_description():
    with st.expander("ℹ️ Tổng quan", expanded=True):
        st.markdown("""
        The dataset contains online retail transaction records from a **:violet[UK-based e-commerce company]**.
        
        **Each transaction includes information such as:**
        * **:blue[Invoice details]** (Hóa đơn) & **:blue[Product info]** (Sản phẩm)
        * **:green[Quantity]** (Số lượng) & **:green[Unit Price]** (Đơn giá)
        * **:red[Customer ID]** (Khách hàng) & **:red[Country]** (Quốc gia)
        
        *These records provide valuable information regarding customer behavior, product demand & revenue generation.*
        """)

def data_overview_show(df):
    c1, _, c2 = st.columns([4.6, 0.2, 4])
    with c1:
        st.markdown(""" #### <span style=color:#88E788> DATA OVERVIEW </span>""", unsafe_allow_html=True)
        st.dataframe(df.head(), hide_index=True)
    with c2:
        data_description()

    st.write("----------------")

# ==================================================================
#  DATA QUALITY & AGG FIGURE
# ==================================================================
def basic_DataQuality(df):
    null_cnt = df.isnull().sum()
    null_perc = (null_cnt / len(df)).apply(lambda x: f"{(100*x):.2f} %" if x > 0 else '0 %')
    n_unique = [df[col].nunique() for col in df.columns]
    
    return pd.DataFrame({
        'count.null': null_cnt,
        'perc_null': null_perc,
        'n_distinct': n_unique
    }).T

def get_quality_table(df):
    quality_df = basic_DataQuality(df)
    st.write("**:violet[Data Quality Table]**")
    styled = (
            quality_df.style
            .set_table_styles([
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#1976D2"),
                        ("color", "white"),
                        ("text-align", "center"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("text-align", "center"),
                    ],
                },
            ])
        )
    st.markdown(
        f'<div style="display:flex;justify-content:center;">{styled.to_html()}</div>',
        unsafe_allow_html=True,
    )

def get_agg_table(df):
    agg_df = df.drop(columns='InvoiceDate').describe()
    for col_name in ['CustomerID', 'Quantity']:
        agg_df[col_name] = agg_df.apply(lambda row: str(int(row[col_name])) if row.name in ['count', 'min', 'max', '25%', '50%', '75%'] 
                                                                            else str(round(row[col_name], 2)), 
                                                axis=1)
    agg_df['UnitPrice'] = agg_df.apply(lambda row: str(int(row['UnitPrice'])) if row.name in ['count', 'max'] 
                                                                            else str(round(row['UnitPrice'], 2)), 
                                                axis=1)
    agg_df = agg_df.T
    st.write("**:violet[Data Aggregation Table (removed duplicates)]**")
    styled = (
        agg_df.style
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#1976D2"),
                    ("color", "white"),
                    ("text-align", "center"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("text-align", "center"),
                ],
            },
        ])
    )
    st.markdown(
        f'<div style="display:flex;justify-content:center;">{styled.to_html()}</div>',
        unsafe_allow_html=True,
    )

def quality_and_metric_show(df, data_dim, memory_usg, n_dup_rows):
    _, c1, _, c2, _ = st.columns([0.2, 4.2, 0.5, 4.5, 0.1])

    with c1:
        st.markdown(""" #### <span style="color: #FD7C6E"> BASIC INFO </span>""", unsafe_allow_html=True  )
        get_quality_table(df)
        get_agg_table(df)

    with c2:
        get_ovv_metric_table(df, data_dim, memory_usg, n_dup_rows)

    st.write("----------------")

# =======================================================================
# METRIC OVERVIEW
# =======================================================================
def get_ovv_metric_table(df, data_dim, memory_usg, n_dup_rows):
    st.markdown(""" #### <span style="color: #FD7C6E"> METRIC OVERVIEW </span>""", unsafe_allow_html=True)

    c1, _, c2, _, c3 = st.columns([1, 0.1, 1, 0.1, 1])
    with c1:
        get_box(icon="📊", value=f"{data_dim[0]} x {data_dim[1]}", title="Init Data Dimensions")
    with c2:
        get_box(icon="💾", value=memory_usg, title="Memory Usage")
    with c3:
        get_box(icon="🧹", value=f"{n_dup_rows} rows", title="Duplicated Rows")

    st.write("### ")

    r1, _, r2, _, r3, _, r4 = st.columns([1.05, 0.08, 1, 0.08, 1, 0.08, 1.65])
    with r1:
        get_box(icon="🛒", value="25900", title="Total Orders")
    with r2:
        get_box(icon="👥", value="4372", title="Total Customers")
    with r3:
        get_box(icon="📦", value="4070", title="Total Products")
    with r4:
        total_rev = (df['UnitPrice'] * df['Quantity']).sum()
        get_box(icon="💰", value=f"$ {total_rev:,.2f}", title="Total Revenue")

def get_box(icon, value, title, change=None, trend=None):
    
    st.markdown(
        f"""
<div style="
background:#414b49;
padding:20px;
border-radius:15px;
">

<div style="font-size:30px">
{icon}
</div>

<div style="font-size:32px;
font-weight:bold;
color:white;">
{value}
</div>

<div style="
display:flex;
justify-content:space-between;
color:white;
">
<span>{title}</span>
<span>{change or ""}</span>
</div>

</div>
""",
        unsafe_allow_html=True
    )

# =======================================================================
# DATA VALIDATION
# =======================================================================
def location_agg(df):
    df = df.sort_values('InvoiceDate').copy()
    df['group'] = (df['Country'] != df['Country'].shift()).cumsum()

    return (df.groupby(['group', 'Country'], as_index=False)
              .agg(start=('InvoiceDate', 'first'),
                   end=('InvoiceDate', 'last'),
                   invoices=('InvoiceNo', 'nunique'),
                   observations=('InvoiceDate', 'size'))
              .assign(
                  start=lambda x: x['start'].dt.strftime('%Y-%m-%d %H:%M'),
                  end=lambda x: x['end'].dt.strftime('%Y-%m-%d %H:%M'),
                  observations=lambda x: x['observations'].astype(str) + ' obs'
              )
              .set_index('group'))

def customer_report(df):
    with st.expander("👤 CustomerID & Description", expanded=True):
        c1, c2 = st.columns([5, 3], border=True)
        with c1:
            st.markdown(
                """
                    <span style="color:red"> **CustomerID** </span> contains a significant proportion of missing values <span style="color:red"> (25.16%) </span>, suggesting many transactions may come from <span style="color:orange"> guest users </span> or <span style="color:#fc9d03"> unidentified customers </span>. This can impact customer-level analyses such as segmentation or lifetime value modeling.

                    <span style="color:cyan"> **Description** </span> has only <span style="color:cyan"> 0.27% </span> missing values and can potentially be recovered from <span style="color:#03adfc"> **StockCode** </span>, indicating a minor data completeness issue.
                """, unsafe_allow_html=True
            )
            st.markdown("""
                            #### <span style="color:#88E788"> Conclusion </span>
                        """, unsafe_allow_html=True )
            st.markdown("""
                    - <span style="color:red"> **CustomerID** </span> exhibits a nearly one-to-one relationship with <span style="color:violet"> **Country** </span>: **4,364** out of **4,372** customers (**99.82%**) are associated with a single country. 
                    - Only **8** customers appear in <span style="color:orange"> multiple countries </span>. Interestingly, these customers alternate between countries instead of permanently <span style="color:lightgreen"> changing location </span>, suggesting temporary shipping destinations or minor metadata inconsistencies rather than systematic data quality issues.

                    => <span style="color:#b3c6e3 "> *Since such cases account for only 0.18% of customers, the overall mapping between CustomerID and Country is considered highly consistent.* </span>
                    """, unsafe_allow_html=True)

        with c2:
            cust_id_wrt_country = df.groupby('CustomerID')['Country'].nunique()
            n_total = cust_id_wrt_country.size
            n_single = (cust_id_wrt_country == 1).sum()
            n_multi = (cust_id_wrt_country > 1).sum()

            summary_df = pd.DataFrame({
                "Notes": [
                    "CustomerID chỉ thuộc 1 quốc gia",
                    "CustomerID thuộc nhiều quốc gia",
                ],
                "Values": [str(n_single), f"{n_multi} ({n_multi/n_total*100:.2f}%)"],
            })
            
            d1, _, d2 = st.columns([5, 1, 3])
            with d1:
                st.dataframe(summary_df.set_index('Notes'), width='stretch')
            with d2:
                formated_idx = [int(idx) for idx in cust_id_wrt_country[cust_id_wrt_country > 1].index]
                cust_id = st.selectbox('Select CID', formated_idx)
            
            st.markdown(f"""
                            ##### <span style="color:#E788E7"> CustomerID = {cust_id} Transaction History </span>
                        """, unsafe_allow_html=True)
            _, c, _ = st.columns([0.1, 20, 0.1])
            with c:
                temp = df.loc[df['CustomerID'] == cust_id, ['InvoiceNo', 'InvoiceDate', 'Country']]
                st.dataframe(location_agg(temp), width='stretch')

def negative_amount_report(df):
    temp = df.copy()
    temp['is_Cancel/Credit'] = temp['InvoiceNo'].astype(str).str.startswith('C')
    temp['is_neg_unit_price'] = temp['UnitPrice'] < 0
    temp['is_neg_quant'] = temp['Quantity'] < 0
    res = temp.groupby(['is_neg_quant', 'is_neg_unit_price', 'is_Cancel/Credit']).size().reset_index()
    res.columns = ['is_neg_quant', 'is_neg_unit_price', 'is_Cancel/Credit', 'count']

    st.dataframe(res, hide_index=True, width='stretch')

    return temp

def duplicate_report(df):
    with st.expander("Others suspicious", expanded=True):
        c1, c2, c3 = st.columns([2.8, 3.2, 4], border=True)
        with c3:
            st.markdown("""
                        ##### <span style="color:#88E788"> Conclusion </span>

                        While most <span style="color:#FF999C"> **negative-quantity** transactions </span> <span style="color:cyan"> correspond to credit notes </span>, <span style="color:#FF999C"> **1,336 records do not** </span>. 

                        - These transactions consistently have zero unit prices and missing customer identifiers, and involve a wide variety of stock codes across multiple dates. This pattern suggests that they are more likely internal stock adjustments or operational records rather than customer purchases.
                        - Therefore, these records should be flagged and handled separately instead of being treated as ordinary sales transactions or simple data errors.
                        """, unsafe_allow_html=True)

            st.markdown("""
                        ##### <span style="color:#E788E7"> Top duplicated combinations </span>
                        """, unsafe_allow_html=True)
            key_cols = ['InvoiceNo', 'StockCode', 'Quantity', 'UnitPrice']
            dup_df = df[df.duplicated(subset=key_cols, keep=False)]
            top_dup = dup_df.groupby(key_cols).size().sort_values(ascending=False).reset_index(name='count').head(5)
            _, c, _ = st.columns([1, 10, 1])
            with c:
                st.dataframe(top_dup.set_index(key_cols).style.set_properties(**{'text-align': 'center'}).set_table_styles([dict(selector='th', props=[('text-align', 'center')])]))

        with c1:
            temp = negative_amount_report(df)
            st.markdown("""
                        - ✅ Có <span style="color:cyan"> **526052** </span> giao dịch mua bán là <span style="color:cyan"> **hoàn toàn bình thường** </span> <span style="color:orange"> (Quantity > 0; Unit_Price > 0 và Invoice không bị **Cancel** hoặc ko phải **Credit**) </span>
                        - ✅ Có <span style="color:violet"> **9251** </span> giao dịch đến từ Credit hoặc Cancel
                        - ❌ <span style="color:#FF999C"> **Nhóm nguy hiểm** </span> <span style="color:#FFC1C3"> (Quantity < 0 nhưng Invoice không phải Credit/Cancel) có đến **1336** cases </span>
                        - ❌ 2 cases cần double check sẽ là nhóm đầu tiên mà ta cần điều tra trước tiên <span style="color:#F45156"> (**UnitPrice < 0**) </span>
                        """, unsafe_allow_html=True)

        with c2:
            st.write("- Khi UnitPrice < 0,")
            st.dataframe(temp.loc[temp['is_neg_unit_price'], ['InvoiceNo','StockCode','Description', 'Quantity','UnitPrice','CustomerID']],
                         hide_index=True, width='stretch')
            st.markdown("""
                    ✅ Như vậy với <span style="color:cyan"> **Description = `Adjust bad debt`** </span> là hoàn toàn hợp lệ. 

                    Tiếp theo, ta sẽ phân tích chuyên sâu hơn  <span style="color:#FF999C"> nhóm bất thường </span> (1336 cases) còn lại, 
                    """, unsafe_allow_html=True)
            # st.write('Missing values trong nhóm 1336 giao dịch này:')            
            c2a, _, c2b = st.columns([3, 0.1, 5])
            with c2a:
                mask = (temp['is_neg_quant']) & (~temp['is_Cancel/Credit'])
                mask_df = temp.loc[mask][['Description', 'CustomerID']].isnull().sum().reset_index()
                mask_df.columns = ['Column', 'n_miss_val']
                st.dataframe(mask_df.set_index('Column'), width='stretch')
            with c2b:
                unit_price_ser = temp.loc[mask, 'UnitPrice'].value_counts()
                st.markdown(f"""
                            
                            - <span style="color:#FF999C"> UnitPrice: </span> value = {unit_price_ser.index[0]} có {unit_price_ser.values[0]} records, chiếm {(100*unit_price_ser.values[0] / 1336)}% toàn bộ nhóm này
                            - <span style="color:cyan"> COUNT DISTINCT (StockCode) </span>: {temp.loc[mask, "StockCode"].nunique()}
                            """, unsafe_allow_html=True)

def intuition(df):
    st.markdown(""" #### <span style="color: cyan"> INTUITIVE DATA VALIDATION </span>""", unsafe_allow_html=True)
    customer_report(df)
    duplicate_report(df)

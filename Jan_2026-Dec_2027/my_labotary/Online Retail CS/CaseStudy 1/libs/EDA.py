import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

bar_block_height = 0.65

# ============================================================================================================================
# REVENUE ANALYSIS
# ============================================================================================================================
def revenue_chart(df):
    avg_RevByOrder = df.groupby('InvoiceNo')['revenue'].sum().mean()

    fig, ax = plt.subplots(1, 2, figsize=(20, 5), gridspec_kw={'width_ratios': [2, 1]}) 
    df.groupby('Date')['revenue'].sum().plot(ax=ax[0], color='#1f77b4', linewidth=1.5)
    ax[0].set_title('Total Revenue (Daily)', fontweight='bold', fontsize=12, pad=10, color='darkgreen')
    ax[0].grid(True, linestyle='--', alpha=0.5)
    ax[0].set_xlabel('Date', color='#008080', fontsize=12, fontweight='bold')
    ax[0].set_ylabel('Revenue ($)', color='#008080', fontsize=12, fontweight='bold')
    ax[0].tick_params(axis='both', labelsize=11, labelcolor='#008080')

    temp = df.copy()
    temp.groupby('Month')['revenue'].sum().plot(marker='o', color='red', linewidth=2, ax=ax[1])
    ax[1].set_title('Trend (Total Revenue) by Month', fontweight='bold', fontsize=12, pad=10, color='darkgreen')
    ax[1].grid(True, linestyle='--', alpha=0.5)
    ax[1].set_xlabel('Month', color='#008080', fontsize=12, fontweight='bold')
    ax[1].set_ylabel('Revenue ($)', color='#008080', fontsize=12, fontweight='bold')
    ax[1].tick_params(axis='both', labelsize=11, labelcolor='#008080')
    ax[1].set_xticks(range(1, 13)) 

    plt.tight_layout()
    st.pyplot(fig)

    return avg_RevByOrder

def revenue_report(df):
    c1, c2 = st.columns([5, 1], border=True)
    with c1:
        avg_RevByOrder = revenue_chart(df)
    with c2:
        st.markdown(f""" 
                        #### **:violet[Tổng kết]**
                    
                        - Tổng doanh thu cao nhất vào các tháng cuối năm
                        - Ngoài U.K (quốc gia sở tại, chiếm {df[df['Country'] == 'United Kingdom']['revenue'].sum() / df['revenue'].sum() * 100:.2f}%) thì Hà Lan là thị trường lớn thứ 2, chiếm {df[df['Country'] == 'Netherlands']['revenue'].sum() / df['revenue'].sum() * 100:.2f}% tổng doanh thu
                        - :red[**Average Order Value**: 💲 {avg_RevByOrder:,.2f} ]
                        - Sản phẩm bán chạy nhất là <span style="color:lightgreen"> "{df.groupby('Description')['revenue'].sum().idxmax()}" </span> với tổng doanh thu là 💲 {df.groupby('Description')['revenue'].sum().max():,.2f} ; chiếm {df.groupby('Description')['revenue'].sum().max() / df['revenue'].sum() * 100:.2f}% tổng doanh thu.
                    """, unsafe_allow_html=True)

# ============================================================================================================================
#  ADVANCED REVENUE CHARTS
# ============================================================================================================================
def other_revenue_charts(df):
    c1, c2 = st.columns([1.5, 5], border=True)
    with c2:
        left, right = st.columns(2)
        with left:
            chart_name = st.selectbox("Select a chart",
                                    ['By Country', 'By Product (revenue)', 'By Product (Quantity Sold)', 'By InvoiceNo'])
        with right:
            top_N = st.slider("Select top N", min_value=5, max_value=20, value=10, step=1)

        fig, html_block = advanced_revenue_chart(df, chart_name, top_N)
        st.pyplot(fig)

        if chart_name == "By InvoiceNo":
            st.markdown(""" 
                            ##### <span style="color: #FD7C6E"> Summary Statistics of Top Revenue Invoices </span>
                        """, unsafe_allow_html=True)
            show_invoiceNo_summary(df)

    with c1:
        st.markdown(html_block, unsafe_allow_html=True)

def advanced_revenue_chart(df, chart_name, top_N):
    if chart_name == "By Country":
        return Revenue_by_Country(df, top_N)
    elif chart_name == "By Product (revenue)":
        return Revenue_by_Product(df, top_N)
    elif chart_name == "By Product (Quantity Sold)":
        return QuantitySold_by_Product(df, top_N)
    elif chart_name == "By InvoiceNo":
        return Revenue_by_InvoiceNo(df, top_N)

def show_invoiceNo_summary(df):
    invoice_summary = (
        df.groupby('InvoiceNo')
        .agg(
            Revenue=('revenue', 'sum'),
            Distinct_Products=('StockCode', 'nunique'),
            Total_Quantity=('Quantity', 'sum')
        )
    )

    invoice_summary['Avg_Item_Price'] = (
        invoice_summary['Revenue'] / invoice_summary['Total_Quantity']
    ).round(2)

    invoice_summary = (
        invoice_summary
            .sort_values('Revenue', ascending=False)
            .head(20)
            .reset_index()
    )
    _, c, _ = st.columns([1, 6, 1])
    with c:
        st.table(
            invoice_summary.set_index('InvoiceNo').style
                .format({
                    'Revenue': '${:,.2f}',
                    'Distinct_Products': '{:,.0f}',
                    'Total_Quantity': '{:,.0f}',
                'Avg_Item_Price': '${:,.2f}'
            })
    )

# Revenue by country =========================================================================================================
def revenue_wrt_country_chart(df, top_N):
    max_height = int(top_N * bar_block_height)
    fig, ax = plt.subplots(1, 2, figsize=(20, max_height))

    agg_rev = df.groupby('Country')['revenue'].sum().sort_values(ascending=False)
    agg_ivc = df.groupby('Country')['InvoiceNo'].nunique().sort_values(ascending=False)
    ans = []

    for idx, col, tit, top10 in zip([0, 1], ['revenue', 'InvoiceNo'], 
                                    [f'Top {top_N} countries có doanh thu cao nhất', f'Top {top_N} countries có số đơn hàng nhiều nhất (n_distinct.invoices)'], 
                                    [agg_rev, agg_ivc]):
        top_perc = 100 * top10.head(top_N).sum() / top10.sum()
        ans.append(top_perc)

        top10 = top10.copy().head(top_N).reset_index()
        sns.barplot(top10, x=col, y='Country', hue='Country', legend=False, ax=ax[idx])
        ax[idx].set_xlim(0, top10[col].max() * 1.25)
        ax[idx].set_title(f"{tit} \n chiếm {top_perc:.2f} % tổng số", fontsize=14, fontweight='bold', pad=15, color='blue')
        ax[idx].grid(True, linestyle='--', alpha=0.5)
        
        for container in ax[idx].containers:
            if idx == 0:
                ax[idx].bar_label(container, fmt=' \${:,.2f}', padding=15, color='blue', 
                                  fontweight='bold', label_type='edge', fontsize=12,
                                  bbox=dict(facecolor='blue', alpha=0.15))
            else:
                ax[idx].bar_label(container, fmt=' {:,.0f}', padding=15, color='blue', 
                                  fontweight='bold', label_type='edge', fontsize=12,
                                  bbox=dict(facecolor='blue', alpha=0.1))

        ax[idx].tick_params(axis='both', labelcolor='blue')
        ax[idx].set_xlabel('REVENUE ($)', fontsize=11, fontweight='bold', color='green')
        ax[idx].set_ylabel('COUNTRY', fontsize=11, fontweight='bold', color='green')

    plt.tight_layout()

    return fig, ans

def Revenue_by_Country(df, top_N=10):
    fig, ans = revenue_wrt_country_chart(df, top_N)
    rev_perc, n_order_perc = ans
    html_block = f""" 
                        ##### **:violet[Key Insights:]**
                        - <span style="color: #FF7276"> **Extreme Market Concentration** </span>: The UK overwhelmingly dominates both total revenue <span style="color:cyan"> (~$8.17M) </span> and order volume <span style="color:cyan">(23,494 invoices)<span style="color:cyan">, accounting for the vast majority of business.
                        - <span style="color: #FF7276"> **Top Impact** </span>: Top <span style="color:orange"> {top_N} countries </span> generate <span style="color:orange"> {rev_perc:.2f}% </span> of revenue and <span style="color:orange"> {n_order_perc:2f}% </span> of total orders.
                        - <span style="color:lightgreen"> Netherlands </span> ranks #2 in revenue <span style="color:cyan">($284k)</span> with just <span style="color:cyan"> 101 </span> orders, indicating a very high AOV.

                        ##### **:violet[Business Actions]**:

                        - <span style="color: #FF7276"> **Protect Core Market** </span>: Prioritize retention and loyalty programs for the UK market.
                        - <span style="color: #FF7276"> **Identify Growth Opportunities** </span>: Evaluate high-basket-value countries like <span style="color:lightgreen"> Netherlands & EIRE </span> (high revenue despite lower order counts) for targeted expansion.                        

                    """
    return fig, html_block

# Revenue by product ==================================================================================
def revenue_wrt_product_chart(df, top_N):
    max_height = int(top_N * bar_block_height)
    text_colors = ['red', 'blue']
    padding = [10, 15]
    fig, ax = plt.subplots(1, 2, figsize=(20, max_height))

    for idx, (cond, color, tit) in enumerate(zip([True, False], ['red', 'blue'], ['LOWEST PRODUCT', 'TOP PRODUCT'])):
        top_prod = df.groupby('Description')['revenue'].sum().sort_values(ascending=cond).head(top_N).reset_index()
        pad = padding[idx]
        color = text_colors[idx]
        sns.barplot(top_prod, x='revenue', y='Description', hue='Description', legend=False, ax=ax[idx])

        # Set limit (cần xử lý nếu có âm)
        max_val = top_prod['revenue'].max()
        min_val = top_prod['revenue'].min()
        ax[idx].set_xlim(min(0, min_val * 1.2), max_val * 1.25)
        
        ax[idx].set_title(f"{tit} REVENUE", fontsize=14, fontweight='bold', pad=15, color=color)
        ax[idx].grid(True, linestyle='--', alpha=0.5)
        
        for container in ax[idx].containers:
            ax[idx].bar_label(container, fmt='\$ {:,.0f}', padding=pad, fontsize=11,
                              color=color, fontweight='bold', label_type='edge',
                              bbox=dict(facecolor=color, alpha=0.1))

        ax[idx].tick_params(axis='both', labelcolor=color)
        ax[idx].set_xlabel('REVENUE', fontsize=11, fontweight='bold', color='green')
        ax[idx].set_ylabel('PRODUCT (Description)', fontsize=11, fontweight='bold', color='green')

    plt.tight_layout()

    return fig

def Revenue_by_Product(df, top_N):
    fig = revenue_wrt_product_chart(df, top_N)
    html_block = f""" 
                    ##### **:violet[Key Insights]**
                    - Low-revenue items are mainly fees, commissions, and discounts <span style="color:#FF7276"> (Amazon Fee, Adjust Bad Debt, Manual, CRUX Commission, Bank Charges) </span>, which reduce overall profitability.
                    - High-revenue products <span style="color:lightgreen"> (DOTCOM POSTAGE, REGENCY CAKESTAND 3 TIER, WHITE HANGING HEART T-LIGHT HOLDER, PARTY BUNTING) </span> show strong customer demand and drive significant revenue.
                    - Analyzing both ends of the spectrum highlights cost drains versus profit drivers.

                    ##### **:violet[Business Actions]**
                    - Monitor and optimize operational costs to minimize negative revenue impact.
                    - Prioritize inventory, marketing, and assortment strategies for top-performing products.
                    - Balance efforts by reducing waste from low-value items while maximizing sales potential of high-demand products.

                    """
    
    return fig, html_block

# Quantity-Sold by product ==================================================================================
def quantity_sold_wrt_product_chart(df, top_N):
    max_height = int(top_N * bar_block_height)
    padding = [10, 15]
    text_colors = ['red', 'blue']

    fig, ax = plt.subplots(1, 2, figsize=(20, max_height))

    # Top invoices highest & lowest
    for idx, (cond, color, tit) in enumerate(zip([True, False], ['red', 'blue'], ['LOWEST PRODUCT', 'TOP PRODUCT'])):
        top_prod = df.groupby('Description')['Quantity'].sum().sort_values(ascending=cond).head(top_N).reset_index()
        sns.barplot(top_prod, x='Quantity', y='Description', hue='Description', legend=False, ax=ax[idx])

        max_val = top_prod['Quantity'].max()
        min_val = top_prod['Quantity'].min()
        ax[idx].set_xlim(min(0, min_val * 1.2), max_val * 1.25)
        
        ax[idx].set_title(f"{tit} QUANTITY SOLD", fontsize=14, fontweight='bold', pad=15, color=color)
        ax[idx].grid(True, linestyle='--', alpha=0.5)
        
        for container in ax[idx].containers:
            ax[idx].bar_label(container, fmt='{:,.0f}', 
                              bbox=dict(facecolor=color, alpha=0.1),
                              padding=padding[idx], color=text_colors[idx], fontweight='bold', label_type='edge')

        ax[idx].tick_params(axis='both', labelcolor=color)
        ax[idx].set_xlabel('QUANTITY SOLD', fontsize=11, fontweight='bold', color='green')
        ax[idx].set_ylabel('PRODUCT (Description)', fontsize=11, fontweight='bold', color='green')

    plt.tight_layout()

    return fig

def QuantitySold_by_Product(df, top_N):
    fig = quantity_sold_wrt_product_chart(df, top_N)
    html_block = f""" 
                    ###### **:violet[Key Findings:]**
                    - <span style="color:lightgreen"> **Best Sellers** </span>: Demand is driven heavily by low-cost/bulk items, led by <span style="color:lightgreen"> WORLD WAR 2 GLIDERS (53.7k units) </span> and <span style="color:lightgreen"> JUMBO BAG RED RETROSPOT (47.2k units) </span>.
                    - <span style="color:FF7276"> **Operational Loss Alert (Left Chart)** </span>: Negative quantity items represent inventory shrinkage, damaged stock, and manual adjustments (e.g., <span style="color:FF7276"> "printing smudges", "unsaleable/destroyed" </span>), NOT actual low-demand sales.
                    
                    ###### **:violet[Actionable Recommendations:]**
                    - <span style="color:lightgreen"> **Inventory Control** </span>: Audit warehouse operations to minimize <span style="color:lightgreen"> damaged/thrown away items (over 34k+ units lost in top entries alone) </span>.
                    - <span style="color:FF7276"> **Data Cleaning** </span>: Next step, <span style="color:FF7276"> exclude </span> system adjustment codes from <span style="color:FF7276"> commercial sales performance analysis </span>.

                    """
    
    return fig, html_block

# Revenue by InvoiceNo ==================================================================================
def revenue_wrt_invoice_chart(df, top_N):
    max_height = int(top_N * bar_block_height)
    text_colors = ['red', 'blue']
    padding = [10, 15]
    
    fig, ax = plt.subplots(1, 2, figsize=(20, max_height))

    # ===========================
    # Left: Top invoices by revenue
    # ===========================
    top_ivc = df.groupby('InvoiceNo')['revenue'].sum().sort_values(ascending=False).head(top_N).reset_index()
    sns.barplot(data=top_ivc, x='revenue', y='InvoiceNo', hue='InvoiceNo', legend=False, ax=ax[0])

    max_val = top_ivc['revenue'].max()
    min_val = top_ivc['revenue'].min()
    ax[0].set_xlim(min(0, min_val * 1.2), max_val * 1.25)
    ax[0].set_title(f"TOP {top_N} INVOICES (REVENUE)", fontsize=14, fontweight='bold', color='blue')
    ax[0].grid(True, linestyle='--', alpha=0.5)

    for container in ax[0].containers:
        ax[0].bar_label(container, fmt='$ {:,.0f}', 
                        padding=padding[0], color=text_colors[0], fontweight='bold',
                        bbox=dict(facecolor=text_colors[0], alpha=0.1), label_type='edge')

    ax[0].tick_params(axis='both', labelcolor=text_colors[0])
    ax[0].set_xlabel('REVENUE ($)', fontsize=11, fontweight='bold', color='green')
    ax[0].set_ylabel('INVOICE NO', fontsize=11, fontweight='bold', color='green')

    # ===========================
    # Right: Top invoices by distinct products
    # ===========================
    top_prod = df.groupby('InvoiceNo')['StockCode'].nunique().sort_values(ascending=False).head(top_N).reset_index(name='n_products')

    sns.barplot(data=top_prod, x='n_products', y='InvoiceNo', hue='InvoiceNo', legend=False, ax=ax[1])
    max_val = top_prod['n_products'].max()
    ax[1].set_xlim(0, max_val * 1.25)
    ax[1].set_title(f"TOP {top_N} INVOICES (DISTINCT PRODUCTS)", fontsize=14, fontweight='bold', color='blue')
    ax[1].grid(True, linestyle='--', alpha=0.5)

    for container in ax[1].containers:
        ax[1].bar_label(container, fmt='%d', 
                        padding=padding[1], color=text_colors[1], fontweight='bold',
                        bbox=dict(facecolor=text_colors[1], alpha=0.1), label_type='edge')

    ax[1].tick_params(axis='both', labelcolor=text_colors[1])
    ax[1].set_xlabel('DISTINCT PRODUCTS', fontsize=11, fontweight='bold', color='green')
    ax[1].set_ylabel('INVOICE NO', fontsize=11, fontweight='bold', color='green')
    plt.tight_layout()

    return fig

def Revenue_by_InvoiceNo(df, top_N):
    fig = revenue_wrt_invoice_chart(df, top_N)
    html_block = f""" 
                    ##### **:violet[Key Insights]**
                    - InvoiceNo 581483 generates the highest revenue ($168,470).
                    - InvoiceNo 573585 has the widest product diversity (1,110 distinct items).
                    - High-revenue invoices do not necessarily overlap with those containing the most diverse product assortments.
                    - Comparing both charts highlights the difference between financial contribution and customer purchasing variety.

                    ##### **:violet[Business Actions]**
                    - Focus retention and upselling strategies on customers tied to top-revenue invoices.
                    - Leverage diverse-product invoices for cross-selling, bundling, and assortment optimization.
                    - Combine insights from both revenue and product diversity to strengthen customer loyalty and maximize long-term growth.
                    
                    """

    return fig, html_block

def basic_EDA_show(df):
    with st.expander("**:violet[Revenue Analysis]**", expanded=True):
        revenue_report(df)
    
    with st.expander("**:violet[Other Reports]**", expanded=True):
        other_revenue_charts(df)

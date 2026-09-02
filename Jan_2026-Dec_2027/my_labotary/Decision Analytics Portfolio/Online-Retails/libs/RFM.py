import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ========================================================================================================
#                                           1. Pareto report
# ========================================================================================================
def get_pareto_df(df):
    product_revenue = df.groupby('Description')['revenue'].sum().sort_values(ascending=False)
    pareto_df = pd.DataFrame(product_revenue)
    pareto_df['CumRev'] = pareto_df['revenue'].cumsum()
    pareto_df['CumPerc'] = (pareto_df['CumRev'] / pareto_df['revenue'].sum()) * 100

    return pareto_df

def processed_pareto_df(df):
    pareto_df = df.copy().drop(columns='revenue')
    top_df = pareto_df.head()
    mid_df = pd.DataFrame([['...'] * len(pareto_df.columns)], 
                          columns=pareto_df.columns, index=['...'])
    tail_df = pareto_df.tail(2)
    final_df = pd.concat([top_df, mid_df, tail_df])

    # Chỉ format những dòng số, bỏ qua dòng '...'
    mask = final_df.index != '...'
    final_df.loc[mask, 'CumRev'] = final_df.loc[mask, 'CumRev'].apply(lambda x: f"$ {x:,.3f}")
    final_df.loc[mask, 'CumPerc'] = final_df.loc[mask, 'CumPerc'].apply(lambda x: f"{x:.3f} %")

    st.dataframe(final_df)

def get_pareto_chart(pareto_df):
    fig, ax = plt.subplots(1,1,figsize=(16, 6))
    plt.plot(range(1, 1+len(pareto_df)), pareto_df['CumPerc'].values, linewidth=2, color="#2525d8")
    plt.axhline(y=80, linestyle='--', color='red')
    plt.axhline(y=0, linestyle='-', color='black')
    plt.axvline(x=760, linestyle='--', color='green', linewidth=1)
    plt.axvline(x=4223, linestyle='--', color='green', linewidth=1)
    plt.title('Pareto Analysis - Product Revenue', color='green', fontsize=18, fontweight='bold')
    plt.ylabel('Cumulative Revenue (%)', color='blue', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Products', color='blue', fontsize=12, fontweight='bold' )
    plt.text(x=800, y=10, s = '760 products', fontsize=14, fontweight='bold', color='#006a4e',
             bbox=dict(facecolor='red', alpha=0.25))
    plt.text(x=3750, y=10, s = 'Total 4223\n products', fontsize=14, fontweight='bold', color='#0000cd',
             bbox=dict(facecolor='green', alpha=0.25))
    plt.text(x=2000, y=85, s = 'approximate 80.03 %', fontsize=15, fontweight='bold',
             bbox=dict(facecolor='blue', alpha=0.15))
    plt.tick_params(axis='both', labelsize=11, labelcolor="#400080")
    plt.xlim(0, 4300)
    plt.grid(True, linestyle='--', alpha=0.2)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f} %'))
    st.pyplot(fig)

def show_pareto_report(pareto_df, thresh = 80):
    cutoff = pareto_df['CumPerc'].ge(thresh).idxmax()
    products_80 = pareto_df.loc[:cutoff]
    # ratio = len(products_80) / len(pareto_df)
    # text_show = f"{len(products_80)} / {len(pareto_df)} products ({ratio:.1%}) generate approximately {thresh} \% of revenue."

    c1, _, c2 = st.columns([2, 0.1, 1])
    with c1:
        get_pareto_chart(pareto_df)
    with c2:
        st.markdown(f"#### <span style='color:#ffa07a'> Pareto report </span>", unsafe_allow_html=True)
        processed_pareto_df(products_80)

def pareto_analysis(df):
    with st.expander("**:violet[Pareto Analysis]**", expanded=True):
        c1, _, c2 = st.columns([1, 0.1, 5])
        with c2:
            get_pareto_df(df)
            show_pareto_report(df)
        with c1:
            st.markdown(f"""
                        #### <span style="color:lightgreen"> Pareto explaination </span>
                        - The Pareto principle, or the 80/20 rule, suggests that roughly 80% of effects come from 20% of causes. In business, this often means that a small number of products or customers generate the majority of revenue.
                        - This insight can guide strategic decisions, such as focusing on high-performing products or optimizing inventory and marketing efforts. 
                        
                        These insights help businesses maximize revenue while minimizing operational complexity.
                        """, unsafe_allow_html=True)

# ========================================================================================================
#                                          2. Customer analytic
# ========================================================================================================
def get_top_customers(df):
    customer_revenue = df.groupby('CustomerID')['revenue'].sum().sort_values(ascending=False)
    customer_frequency = df.groupby('CustomerID')['InvoiceNo'].nunique().sort_values(ascending=False)

    return customer_revenue, customer_frequency

def plot_customer_stats(top_revenue, top_frequency):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4),
                           gridspec_kw={'width_ratios': [7, 4]})
    sns.histplot(top_frequency, bins=50, ax=ax[1])
    ax[1].set_title('Customer Purchase Frequency', color='green', fontsize=15, fontweight='bold')
    ax[1].set_xlabel('Number of Orders', color='blue', fontsize=12, fontweight='bold')
    ax[1].set_ylabel('Number of Customers', color='blue', fontsize=12, fontweight='bold')
    ax[1].tick_params(axis='both', labelsize=10, labelcolor="#400080")
    ax[1].grid(True, linestyle='--', alpha=0.5)

    top_revenue.index = top_revenue.index.astype(int)  # Convert index to string for better display
    top_revenue.head(10).sort_values(ascending=True).plot(kind='barh', ax=ax[0])
    ax[0].set_title('Top 10 Customers by Revenue', color='green', fontsize=15, fontweight='bold')
    ax[0].set_xlabel('Revenue', color='blue', fontsize=12, fontweight='bold')
    ax[0].set_ylabel('CustomerID', color='blue', fontsize=12, fontweight='bold')
    ax[0].tick_params(axis='both', labelsize=10, labelcolor="#400080")
    ax[0].set_xlim(0, top_revenue.max() * 1.2)  # Set x-axis limit to 10% more than max revenue
    ax[0].xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax[0].grid(True, linestyle='--', alpha=0.5)
    for container in ax[0].containers:
        ax[0].bar_label(container, fmt='\$ {:,.0f}', padding=10, fontsize=11,
                            color='#0014a8', fontweight='bold', label_type='edge',
                            bbox=dict(facecolor='cyan', alpha=0.1))

    plt.tight_layout()
    st.pyplot(fig)

def customer_lifetime_plot(df):
    customer_lifetime = df.groupby('CustomerID')['revenue'].sum()
    fig, ax = plt.subplots(figsize=(12, 5.5))   # tạo ax rõ ràng
    sns.histplot(customer_lifetime, bins=80, ax=ax)
    ax.set_title('Customer Lifetime Revenue Distribution',
                 fontsize=14, fontweight='bold', color='green')
    ax.set_xlabel('Lifetime Revenue', fontsize=12, fontweight='bold', color='blue')
    ax.set_ylabel('Number of Customers', fontsize=12, fontweight='bold', color='blue')    
    ax.tick_params(axis='both', labelsize=10, labelcolor="#400080")
    # chỉnh format tick labels trên trục x vs y
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.grid(True, linestyle='--', alpha=0.5)

    return customer_lifetime, fig

def get_customer_lifetime_summary(df):
    st.markdown(f"##### <span style='color:#66cdaa'> Customer Lifetime Revenue Summary </span>", unsafe_allow_html=True)
    desc = df.describe().to_frame().T
    format_dict = {col: '{:,.0f}' if col == 'count' else '{:,.2f}' for col in desc.columns}
    st.dataframe(desc.style.format(format_dict))
    avg_lifetime_revenue = df.mean()
    std_lifetime_revenue = df.std()
    st.markdown(f"""
                    - <span style='color:#ff7f50'> AVG(Customer Lifetime Revenue): </span> 
                    <span style='color:#f08080'> **{avg_lifetime_revenue:,.2f}** </span>
                    - <span style='color:#40e0d0'> STD(Customer Lifetime Revenue): </span> 
                    <span style='color:#adff2f'> **{std_lifetime_revenue:,.2f}** </span>
                """, unsafe_allow_html=True)

def get_top_customer_summary(df, top_revenue):
    total_revenue = df['revenue'].sum()
    st.markdown(f"""
                    ##### <span style='color:#1e90ff'> Top Customer Revenue </span>
                    <span style='color:#ff7f50'> Total revenue : </span> **<span style='color:#f08080'> \$ {total_revenue:,.2f} </span>** <br>
                """, unsafe_allow_html=True)
    top_N_customers = [10, 20, 50, 100, 200, 500]
    total_rev_by_top_CID = {N: top_revenue.head(N).sum() for N in top_N_customers}
    top_rev_by_CID_df = pd.DataFrame(list(total_rev_by_top_CID.items()), columns=['Top N Customers', 'Total Revenue']).set_index('Top N Customers')
    top_rev_by_CID_df['TotalRev_Perc'] = top_rev_by_CID_df['Total Revenue'] / total_revenue * 100
    st.dataframe(top_rev_by_CID_df.style.format({'Total Revenue': '${:,.2f}', 'TotalRev_Perc': '{:.2f}%'}))

def init_summary():
    st.markdown(f"""
                    ##### <span style='color:#66cdaa '> Summary </span>
                    <span style='color:#ffa07a'> N distinct customers: </span> <span style='color:#f08080'> **4372** </span> <br>
                    A larger customer base generally indicates broader market reach and stronger revenue diversification
                    - The top 10 customers by revenue and frequency are identified, providing insights into the most valuable and engaged customers.
                    - This information can inform targeted marketing strategies, loyalty programs, and personalized offers to enhance customer retention and maximize revenue.
                """, unsafe_allow_html=True)

def customer_analytic(df):
    with st.expander("**:violet[Customer Analytic]**", expanded=True):
        top_revenue, top_frequency = get_top_customers(df)
        c1, _, c2, _, c3 = st.columns([1.5, 0.1, 3, 0.1, 1])
        with c1:
            init_summary()
        with c2:
            plot_customer_stats(top_revenue, top_frequency)
        with c3:
            st.markdown(f"##### <span style='color:#1e90ff'> Top Customer Orders </span>", unsafe_allow_html=True)
            customer_orders = df.groupby(['CustomerID','InvoiceNo'])['revenue'].sum()
            st.dataframe(customer_orders.head().apply(lambda x: f"$ {x:,.2f}"))
            AOV = customer_orders.mean()
            st.markdown(f"<span style='color:#ff7f50'> Average Order Value (AOV):</span> **<span style='color:#f08080'> \$ {AOV:.2f} </span>**", unsafe_allow_html=True)
        st.write("---")
        r1, _, r2 = st.columns([4, 0.1, 1])
        with r1:
            customer_lifetime, fig = customer_lifetime_plot(df)
            left, right = st.columns([1, 1], border=True)
            with left:
                # st.markdown(f"##### <span style='color:#66cdaa'> Customer Lifetime Revenue Distribution </span>", unsafe_allow_html=True)
                st.pyplot(fig)
            with right:
                get_customer_lifetime_summary(customer_lifetime)
        with r2:
            get_top_customer_summary(df, top_revenue)

# ========================================================================================================
#                                       3. RFM 
# ========================================================================================================
def get_RFM_table(df):
    snapshot_date = (df['InvoiceDate'].max() + pd.Timedelta(days=1))
    rfm = df.groupby('CustomerID').agg({
                                        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
                                        'InvoiceNo': 'nunique',
                                        'revenue': 'sum'
                                    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    return rfm

def rfm_definition():
    st.markdown("""
                ##### <span style='color:#FFD700'> RFM Explanation </span>

                RFM Analysis is one of the most widely used <span style='color:#00FFFF'>customer segmentation techniques</span> in <span style='color:#ADFF2F'>marketing</span> and <span style='color:#FF69B4'>business analytics</span>.

                - <span style='color:#FFA500'>Recency</span> → How recently a customer made a purchase  
                - <span style='color:#00FF7F'>Frequency</span> → How often a customer makes purchases  
                - <span style='color:#FF4500'>Monetary</span> → How much money a customer spends  

                By analyzing these three dimensions, businesses can identify <span style='color:#00CED1'>valuable customer segments</span> and develop <span style='color:#7CFC00'>targeted retention</span> and <span style='color:#FF6347'>marketing strategies</span>.
    """, unsafe_allow_html=True)

def rfm_distribution_plot(rfm):
    fig, axes = plt.subplots(3, 3, figsize=(18, 9), gridspec_kw={'height_ratios': [3, 2, 3]})
    axes = axes.ravel()
    
    colors = ['violet', 'green', 'red']

    # Recency classes
    recencydf = pd.DataFrame({
        'class': ['(0, 30] days', '(30, 90] days', '(90, 270] days', '(270, ∞) days'],
        'percentage': [
            rfm[rfm["Recency"] <= 30].shape[0]/rfm.shape[0]*100,
            rfm[(rfm["Recency"] > 30) & (rfm["Recency"] <= 90)].shape[0]/rfm.shape[0]*100,
            rfm[(rfm["Recency"] > 90) & (rfm["Recency"] <= 270)].shape[0]/rfm.shape[0]*100,
            rfm[rfm["Recency"] > 270].shape[0]/rfm.shape[0]*100
        ]
    }).set_index('class').sort_values(by='percentage', ascending=True)

    # Frequency classes
    freq_df = pd.DataFrame({
        'class': ['(0, 3] orders', '(3, 10] orders', '(10, ∞) orders'],
        'percentage': [
            rfm[rfm["Frequency"] <= 3].shape[0]/rfm.shape[0]*100,
            rfm[(rfm["Frequency"] > 3) & (rfm["Frequency"] <= 10)].shape[0]/rfm.shape[0]*100,
            rfm[rfm["Frequency"] > 10].shape[0]/rfm.shape[0]*100
        ]
    }).set_index('class').sort_values(by='percentage', ascending=True)

    # Monetary classes
    moneydf = pd.DataFrame({
        'class': ['(0, 1000]', '(1000, 5000]', '(5000, ∞)'],
        'percentage': [
            rfm[rfm["Monetary"] <= 1000].shape[0]/rfm.shape[0]*100,
            rfm[(rfm["Monetary"] > 1000) & (rfm["Monetary"] <= 5000)].shape[0]/rfm.shape[0]*100,
            rfm[rfm["Monetary"] > 5000].shape[0]/rfm.shape[0]*100
        ]
    }).set_index('class').sort_values(by='percentage', ascending=True)

    grouped_dfs = [recencydf, freq_df, moneydf]
    tuples = [('Recency', 'Monetary'), ('Frequency', 'Monetary'), ('Recency', 'Frequency')]

    for i, col in enumerate(rfm.columns):
        # Histograms cho Recency, Frequency, Monetary
        sns.histplot(rfm[col], bins=30, ax=axes[i], color=colors[i % len(colors)])
        axes[i].set_title(f'{col} Distribution', fontsize=14, fontweight='bold', color='green')
        axes[i].set_xlabel(col, fontsize=12, fontweight='bold', color='blue')
        axes[i].set_ylabel('Number of Customers', fontsize=12, fontweight='bold', color='blue')
        axes[i].tick_params(axis='both', labelsize=10, labelcolor="#400080")
        axes[i].grid(True, linestyle='--', alpha=0.5)

        # Bar plots by category segments
        grouped_dfs[i].plot(kind='barh', ax=axes[i + 3], legend=False, color=colors[i % len(colors)])
        for container in axes[i + 3].containers:
            axes[i + 3].bar_label(container, fmt='{:,.2f} %', padding=15, color='blue', 
                                fontweight='bold', label_type='edge', fontsize=12,
                                bbox=dict(facecolor=colors[i % len(colors)], alpha= 0.3 * container.datavalues[0]/grouped_dfs[i]['percentage'].max())) 
        axes[i + 3].tick_params(axis='both', labelsize=10, labelcolor="#400080")
        axes[i + 3].grid(True, linestyle='--', alpha=0.5)
        axes[i + 3].set_xlim(0, 1.5 * grouped_dfs[i]['percentage'].max())

        # Scatter plots for pairwise relationships
        axes[i + 6].scatter(rfm[tuples[i][0]], rfm[tuples[i][1]], color=colors[i % len(colors)], alpha=0.6, s=25)
        axes[i + 6].tick_params(axis='both', labelsize=10, labelcolor="#400080")
        axes[i + 6].set_xlabel(tuples[i][0], fontsize=12, fontweight='bold', color='blue')
        axes[i + 6].set_ylabel(tuples[i][1], fontsize=12, fontweight='bold', color='blue')
        axes[i + 6].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    st.pyplot(fig)

def rfm_distribution_plot_v2(rfm):
    fig, axes = plt.subplots(2, 3, figsize=(18, 6.5), gridspec_kw={'height_ratios': [3, 2]})
    axes = axes.ravel()    
    colors = ['violet', 'green', 'red']

    # Recency classes
    recencydf = pd.DataFrame({
        'class': ['(0, 30] days', '(30, 90] days', '(90, 270] days', '(270, ∞) days'],
        'percentage': [
            rfm[rfm["Recency"] <= 30].shape[0]/rfm.shape[0]*100,
            rfm[(rfm["Recency"] > 30) & (rfm["Recency"] <= 90)].shape[0]/rfm.shape[0]*100,
            rfm[(rfm["Recency"] > 90) & (rfm["Recency"] <= 270)].shape[0]/rfm.shape[0]*100,
            rfm[rfm["Recency"] > 270].shape[0]/rfm.shape[0]*100
        ]
    }).set_index('class').sort_values(by='percentage', ascending=True)

    # Frequency classes
    freq_df = pd.DataFrame({
        'class': ['(0, 3] orders', '(3, 10] orders', '(10, ∞) orders'],
        'percentage': [
            rfm[rfm["Frequency"] <= 3].shape[0]/rfm.shape[0]*100,
            rfm[(rfm["Frequency"] > 3) & (rfm["Frequency"] <= 10)].shape[0]/rfm.shape[0]*100,
            rfm[rfm["Frequency"] > 10].shape[0]/rfm.shape[0]*100
        ]
    }).set_index('class').sort_values(by='percentage', ascending=True)

    # Monetary classes
    moneydf = pd.DataFrame({
        'class': ['(0, 1000]', '(1000, 5000]', '(5000, ∞)'],
        'percentage': [
            rfm[rfm["Monetary"] <= 1000].shape[0]/rfm.shape[0]*100,
            rfm[(rfm["Monetary"] > 1000) & (rfm["Monetary"] <= 5000)].shape[0]/rfm.shape[0]*100,
            rfm[rfm["Monetary"] > 5000].shape[0]/rfm.shape[0]*100
        ]
    }).set_index('class').sort_values(by='percentage', ascending=True)

    grouped_dfs = [recencydf, freq_df, moneydf]

    for i, col in enumerate(rfm.columns):
        # Histograms cho Recency, Frequency, Monetary
        sns.histplot(rfm[col], bins=30, ax=axes[i], color=colors[i % len(colors)])
        axes[i].set_title(f'{col} Distribution', fontsize=14, fontweight='bold', color='green')
        axes[i].set_xlabel(col, fontsize=12, fontweight='bold', color='blue')
        axes[i].set_ylabel('Number of Customers', fontsize=12, fontweight='bold', color='blue')
        axes[i].tick_params(axis='both', labelsize=10, labelcolor="#400080")
        axes[i].grid(True, linestyle='--', alpha=0.5)

        # Bar plots by category segments
        grouped_dfs[i].plot(kind='barh', ax=axes[i + 3], legend=False, color=colors[i % len(colors)])
        for container in axes[i + 3].containers:
            axes[i + 3].bar_label(container, fmt='{:,.2f} %', padding=15, color='blue', 
                                fontweight='bold', label_type='edge', fontsize=12,
                                bbox=dict(facecolor=colors[i % len(colors)], alpha= 0.3 * container.datavalues[0]/grouped_dfs[i]['percentage'].max())) 
        axes[i + 3].tick_params(axis='both', labelsize=10, labelcolor="#400080")
        axes[i + 3].grid(True, linestyle='--', alpha=0.5)
        axes[i + 3].set_xlim(0, 1.5 * grouped_dfs[i]['percentage'].max())
    
    plt.tight_layout()
    st.pyplot(fig)

def rfm_show(rfm):
    with st.expander("**:violet[RFM Analysis]**", expanded=True):        
        c1, c2 = st.columns([1, 3], border=True)
        with c1:
            rfm_definition()
            st.markdown(f"""
                            ###### <span style='color:#66cdaa'> RFM Table </span>
                        """, unsafe_allow_html=True)
            st.dataframe(rfm.head())

        with c2:
            rfm_distribution_plot_v2(rfm)
            left, mid, right = st.columns(3, border=True)
            with left:
                st.markdown(f"""
                                ###### <span style='color:#00bfff'> Recency Summary </span>

                                - Measures the number of days since a customer's most recent purchase.
                                - Lower values indicate more active customers.

                                <span style='color:#00bfff'> **Recency** </span> is right-skewed, indicating that most customers have purchased recently, while a smaller group has been inactive for an extended period.

                            """, unsafe_allow_html=True)
            with mid:
                st.markdown(f"""
                                ###### <span style='color:#32CD32'> Frequency Summary </span>

                                - Measures how often a customer makes purchases.
                                - Higher values indicate more loyal customers.

                                <span style='color:#32CD32'> **Frequency** </span> is highly right-skewed, suggesting that most customers place only a few orders, whereas a limited number of loyal customers purchase frequently.

                            """, unsafe_allow_html=True)

            with right:
                st.markdown(f"""
                                ###### <span style='color:#f08080'> Monetary Summary </span>

                                - Measures the total amount of money a customer has spent.
                                - Higher values indicate more valuable customers.

                                <span style='color:#f08080'> **Monetary** </span> also exhibits a highly skewed distribution, where the majority of customers contribute relatively low spending, while a small proportion of high-value customers account for a significant share of total revenue.

                            """, unsafe_allow_html=True)

def compute_rfm_scores(rfm):
    # Calculate RFM scores using quantiles
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 
                             4, labels=[4,3,2,1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 
                             4, labels=[1,2,3,4])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'],
                             4, labels=[1,2,3,4])

    # Combine RFM scores into a single RFM score
    rfm['RFM_Score'] = (rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str))

    return rfm

# Define customer segments based on RFM scores
def segment_map(row):
    r = row['R_Score']
    f = row['F_Score']

    if r == 4 and f == 4:
        return 'Champions'
    elif r >= 3 and f >= 3:
        return 'Loyal Customers'
    elif r >= 3 and f <= 2:
        return 'Need Attention'
    elif r <= 2 and f >= 3:
        return 'At Risk'
    elif r <= 2 and f <= 2:
        return 'Hibernating'
    else:
        return 'Others'

def stats_summary(df):
    segment_counts = df['Segment'].value_counts()
    segment_revenue = df.groupby('Segment')['Monetary'].sum().sort_values(ascending=True)

    # Consistent color mapping across charts
    segment_colors = {
            'Champions': "#61E79B",        # Dark green
            'Loyal Customers': "#6296CE",  # Blue
            'Need Attention': "#CEB875",   # Yellow
            'At Risk': "#D9AA80",          # Orange
            'Hibernating': "#DF5959"       # Red
        }

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={'width_ratios': [1, 2]}, constrained_layout=True)
    pie_colors = [segment_colors.get(segment, '#BDBDBD') for segment in segment_counts.index]
    pie_labels = [f'{segment} ({count:,})' for segment, count in segment_counts.items()]
    ax[0].pie(segment_counts, labels=pie_labels, labeldistance=1.08, pctdistance=0.72, 
            autopct='%1.1f%%', startangle=90, colors=pie_colors, wedgeprops=dict(edgecolor='white', linewidth=2))
    ax[0].set_title('Customer Segments Distribution', fontsize=12, fontweight='bold', color='green', pad=20)
    ax[0].set_aspect('equal')

    bar_colors = [segment_colors.get(segment, '#BDBDBD') for segment in segment_revenue.index]
    segment_revenue.plot(kind='barh', ax=ax[1], color=bar_colors)
    ax[1].set_title('Revenue Contribution by Segment', fontsize=12, fontweight='bold', color='green')
    ax[1].set_xlabel('Revenue')
    ax[1].set_ylabel('')
    for container in ax[1].containers:
        ax[1].bar_label(container, fmt='$ {:,.2f}', padding=2, fontweight='bold', label_type='edge', fontsize=9)
    ax[1].tick_params(axis='both', labelsize=8, labelcolor="#400080")
    ax[1].set_xlim(0, 1.29 * segment_revenue.max())
    ax[1].grid(True, axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)

def scatterplot_wrt_cust_segmentation(rfm):
    segment_colors = {
        'Champions': "#61E79B",
        'Loyal Customers': "#6296CE",
        'Need Attention': "#CEB875",
        'At Risk': "#D9AA80",
        'Hibernating': "#DF5959"
    }

    tuples = [('Recency', 'Monetary'), ('Frequency', 'Monetary'), ('Recency', 'Frequency')]
    fig, axes = plt.subplots(1, 3, figsize=(18, 3.5))

    for idx, (x_col, y_col) in enumerate(tuples):
        sns.scatterplot(data=rfm, x=x_col, y=y_col, hue='Segment', palette=segment_colors, ax=axes[idx], legend=(idx == 0))
        axes[idx].tick_params(axis='both', labelsize=10, labelcolor="#400080")
        axes[idx].set_xlabel(x_col, fontsize=12, fontweight='bold', color='blue')
        axes[idx].set_ylabel(y_col, fontsize=12, fontweight='bold', color='blue')
        axes[idx].grid(True, linestyle='--', alpha=0.5)

    axes[0].legend(title='Segment', frameon=False)
    fig.tight_layout()
    st.pyplot(fig)

def customer_segmentation(rfm):
    rfm = compute_rfm_scores(rfm)
    rfm['Segment'] = rfm.apply(segment_map, axis=1)

    with st.expander("**:violet[Customer Segmentation]**", expanded=True):
        c1, c2 = st.columns([1, 5], border=True)
        with c1:
            st.markdown(f"""
                            ###### <span style='color:#66cdaa'> RFM score explaination </span>
                        """, unsafe_allow_html=True)

        with c2:
            stats_summary(rfm)
            # push scatter plot here after we getting segmentations result (delete scatterplot above)
            scatterplot_wrt_cust_segmentation(rfm)

# ============================== Time series analysis =========================================================
def get_monthly_trend(df):
    monthly_sales = df.groupby(pd.Grouper(key='InvoiceDate', freq='M'))['revenue'].sum().reset_index()
    monthly_sales['MA_2'] = monthly_sales['revenue'].rolling(2).mean()
    monthly_sales['MA_3'] = monthly_sales['revenue'].rolling(3).mean()
    monthly_sales['GrowthRate'] = monthly_sales['revenue'].pct_change() * 100

    return monthly_sales

def Rev_Trend_MA(df):
    fig, ax = plt.subplots(2, 1, figsize=(16, 6))
    title_color = 'green'
    tick_color = "#9F34C3"

    # Revenue Trend
    ax[0].plot(df['InvoiceDate'], df['revenue'], label='revenue')
    ax[0].plot(df['InvoiceDate'], df['MA_2'], label='2 Month Moving Average')
    ax[0].plot(df['InvoiceDate'], df['MA_3'], label='3 Month Moving Average')
    ax[0].set_title('Revenue Trend with Moving Average', fontsize=12, fontweight='bold', color=title_color)
    ax[0].set_ylabel('revenue', color=tick_color)
    ax[0].tick_params(axis='both', colors=tick_color)
    ax[0].grid(True, linestyle='--', alpha=0.4)
    ax[0].legend(frameon=False)

    # Revenue Growth
    ax[1].plot(df['InvoiceDate'], df['GrowthRate'], label='MoM Growth')
    ax[1].axhline(y=0, linestyle='--', linewidth=1)
    ax[1].set_title('Month-over-Month Revenue Growth', fontsize=12, fontweight='bold', color=title_color)
    ax[1].set_ylabel('Growth Rate (%)', color=tick_color)
    ax[1].tick_params(axis='both', colors=tick_color)
    ax[1].grid(True, linestyle='--', alpha=0.4)
    ax[1].legend(frameon=False)

    fig.tight_layout()
    st.pyplot(fig)

def sale_forecasting(df):
    with st.expander("**:violet[Revenue Trend Report]**", expanded=True):
        c1, c2 = st.columns([1, 4])
        with c2:
            Rev_Trend_MA(df)

    with st.expander("**:violet[Sale Forecasting]**", expanded=True):
        pass

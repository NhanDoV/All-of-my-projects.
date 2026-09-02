import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def repl_eda(df):
    # Tầng 1
    with st.expander("**:red[1. Overview Distribution analysis]**", expanded=True):
        cols = ['on_hand_units', 'on_order_units', 'safety_stock_units']
        df_long = df.melt(id_vars=['sku_id', 'warehouse'], value_vars=cols, var_name='metric', value_name='units')

        # Một figure duy nhất, facet theo metric
        fig = px.box(
            df_long, x='sku_id', y='units', height=420,
            color='warehouse', color_discrete_map={'WH_A': "#238CA1", 'WH_B': "#D9916F"}, 
            facet_col='metric',          # hoặc facet_row='metric' nếu muốn xếp dọc
            title='Distribution of Inventory Metrics by SKU',
            category_orders={'metric': cols}   # giữ thứ tự cột gốc
        )

        # Tối ưu legend → mid-top
        fig.update_layout(
            title_font_color="#B14119",
            legend=dict(
                orientation='h',          # ngang
                yanchor='bottom',
                y=1.08,                   # đẩy lên trên title một chút
                xanchor='center', x=0.5
            ),
            margin=dict(t=100)            # tăng margin top để legend không bị đụng title
        )
        fig.update_yaxes(matches=None)

        # Ẩn title facet mặc định (vì đã có title chung)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
        st.plotly_chart(fig, width='stretch')

        # ------------------------
        param_col, ts_col, fr_col = st.columns([1, 5, 3])
        with param_col:
            st.write("**:green[2. Selection Parameters]**")
            sku_sel = st.selectbox("Select SKU-ID", [f"SKU{i:03d}" for i in range(1, 21)], key="sku_select_tab2")
            wh_sel = st.selectbox("Select Warehouse", ["WH_A", "WH_B"], key="wh_select_tab2")
            filtered_df = df[(df['sku_id'] == sku_sel) & (df['warehouse'] == wh_sel)]

        with ts_col:
            fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
            fig_ts.add_trace(
                go.Scatter(x=filtered_df['date'], y=filtered_df['on_hand_units'],
                        name='on_hand_units', mode='lines'),
                secondary_y=False
            )
            fig_ts.add_trace(
                go.Scatter(x=filtered_df['date'], y=filtered_df['on_order_units'],
                        name='on_order_units', mode='lines'),
                secondary_y=False
            )
            fig_ts.add_trace(
                go.Scatter(x=filtered_df['date'], y=filtered_df['safety_stock_units'],
                        name='safety_stock_units', mode='lines', line=dict(dash='dot')),
                secondary_y=True
            )
            fig_ts.update_layout(
                title=f"Time Series of Inventory Metrics for {sku_sel} at {wh_sel}",
                height=420
            )
            fig_ts.update_yaxes(title_text="on_hand / on_order", secondary_y=False)
            fig_ts.update_yaxes(title_text="safety_stock", secondary_y=True)
            fig_ts.update_layout(
                legend=dict(
                    orientation="h",          # nằm ngang
                    yanchor="bottom",
                    y=1.02,                   # đẩy lên trên plot
                    xanchor="center",
                    x=0.5                     # căn giữa
                ),
                margin=dict(t=80)             # tăng margin top nếu bị đụng title
            )
            st.plotly_chart(fig_ts, width='stretch')

        with fr_col:
            # Boxplot Fill rate by policies
            fig_fr = px.box(
                filtered_df, x='policy', y='fill_rate', color='policy',
                color_discrete_map={'base_stock': "#238CA1", 'minmax': "#D9916F", 'ml_reorder': "#2BB119"},
                title='Fill Rate Distribution by Policy and Warehouse'
            )
            st.plotly_chart(fig_fr, width='stretch')

        # pivot to get stockout by all sku-id & policies
        st.markdown("**:green[3. Stockout Distribution by all SKU-ID and Policy]**")
        stockout_pivot = pd.pivot_table(df, columns='sku_id', index=['warehouse', 'policy'], 
                                        values='stockout', aggfunc='sum', fill_value=0,
                                        margins=True, margins_name='Total')
        st.table(stockout_pivot)

        fillrate_pivot = pd.pivot_table(df, columns='sku_id', index=['warehouse', 'policy'], 
                                        values='fill_rate', aggfunc='mean', fill_value=0)
        st.markdown("**:green[4. Fill Rate Distribution by all SKU-ID and Policy]**")
        st.table(fillrate_pivot.style.format("{:.2%}").highlight_max(axis=1, color='lightgreen').highlight_min(axis=1, color='lightcoral'))

def repl_optimization(df):
    c1, c2 = st.columns([1, 1])
    with c1:
        with st.expander("**:orange[Reorder Point (ROP) & EOQ Calculation]**", expanded=True):
            left, right = st.columns(2, gap = 'medium')
            with left:
                sku_sel = st.selectbox("Select SKU-ID", sorted(df['sku_id'].unique()), key="rop_sku")
                wh_sel = st.selectbox("Select Warehouse", ["WH_A","WH_B"], key="rop_wh")

            filtered_df = df[(df['sku_id']==sku_sel) & (df['warehouse']==wh_sel)]

            # ROP = avg demand × lead_time + safety_stock
            avg_demand = filtered_df['demand_units'].mean()
            lead_time = filtered_df['lead_time_days'].mean()
            safety_stock = filtered_df['safety_stock_units'].mean()
            rop = avg_demand * lead_time + safety_stock

            # EOQ (nếu có cost breakdown: setup cost & holding cost)
            # giả sử có cột order_cost_usd và holding_cost_usd
            if 'order_cost_usd' in df.columns and 'holding_cost_usd' in df.columns:
                D = filtered_df['demand_units'].sum()   # annual demand
                S = filtered_df['order_cost_usd'].mean()
                H = filtered_df['holding_cost_usd'].mean()
                eoq = ((2*D*S)/H)**0.5
            else:
                eoq = None
            with left:
                st.metric("Reorder Point (ROP)", f"{rop:.0f} units")
                if eoq:
                    st.metric("Economic Order Quantity (EOQ)", f"{eoq:.0f} units")

                st.markdown("""
                    <div style="
                        background: linear-gradient(90deg, #e8f5e9, #f1f8e9);
                        border: 1px solid #a5d6a7;
                        padding: 14px 18px;
                        border-radius: 8px;
                        font-size: 16px;
                        color: #1b5e20;
                        text-align: center;
                        margin: 10px 0;
                    ">
                        <span style="font-weight: 600;">ROP</span> = 
                        (average demand × lead_time_days) + safety_stock_units
                    </div>
                """, unsafe_allow_html=True)

            # Bảng ROP/EOQ cho tất cả SKU/warehouse
            rop_table = df.groupby(['sku_id','warehouse']).apply(
                lambda g: (g['demand_units'].mean()*g['lead_time_days'].mean() + g['safety_stock_units'].mean())
            ).reset_index(name='ROP')

            with right:
                st.write("**ROP by SKU & Warehouse**")
                st.table(rop_table.pivot(index='sku_id', columns='warehouse', values='ROP').round(0))

    with c2:
        with st.expander("**:orange[Scenario Simulation]**", expanded=True):
            st.write("Nhập các tham số để mô phỏng…")

            # Input service level target
            service_level_target = st.slider("Service Level Target (%)", 80, 99, 95)
            z_value = 1.65 if service_level_target==95 else 2.33  # ví dụ: 95% ~ 1.65, 99% ~ 2.33

            # Tính safety stock cần thiết
            demand_std = filtered_df['demand_units'].std()
            lead_time = filtered_df['lead_time_days'].mean()
            safety_stock_needed = z_value * demand_std * (lead_time**0.5)

            st.metric("Required Safety Stock", f"{safety_stock_needed:.0f} units")

            # Input budget constraint
            budget = st.number_input("Budget Constraint (USD)", value=1000000, step=50000)
            cost_by_policy = df.groupby('policy')['total_cost_usd'].sum().reset_index()

            st.write("**Policy Costs vs Budget**")
            fig = px.bar(cost_by_policy, x='policy', y='total_cost_usd',
                        color='policy', title="Total Cost by Policy")
            # highlight vượt ngưỡng
            fig.add_hline(y=budget, line_dash="dash", line_color="red", annotation_text="Budget Limit")
            st.plotly_chart(fig, width='stretch')

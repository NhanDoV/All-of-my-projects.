import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def demand_forecast_eda(df):
    df = df.copy()
    df['demand_diff'] = (df['demand_units'] - df['demand_forecast_units'])
    sku_id_list = sorted(df['sku_id'].unique())
    with st.expander("**:red[DEMAND_UNITS]**", expanded=True):
        c1, c2 = st.columns([1, 6], gap="large")
        with c1:
            st.metric("AVG(MAE) all sku & wh", border=True,
                      value=f"{  mean_absolute_error(df['demand_units'], df['demand_forecast_units']):.2f}")
            st.write("**:red[Top 5 highest SKUs_id]**")
            sku_top5 = df.groupby('sku_id')['demand_units'].sum().nlargest(5)
            st.table(sku_top5)

            sku_sel = st.selectbox("Select SKU-ID", sku_id_list)

            # compute based on different of demand_units and demand_forecast_units
            top_under = df.groupby('sku_id')['demand_diff'].sum().nlargest(5)
            top_over = df.groupby('sku_id')['demand_diff'].sum().nsmallest(5)
            st.write("**:red[Top 5 SKU dự đoán bị hụt]**")
            st.table(top_under.to_frame().style.format("{:.2f}"))
            st.write("**:red[Top 5 SKU dự đoán bị thừa]**")
            st.table(top_over.to_frame().style.format("{:.2f}"))

            wh_sel = st.selectbox("Select Warehouse (exclude chart 1)", ["WH_A", "WH_B"])

        with c2:
            piv = pd.pivot_table(df, columns='sku_id', index='warehouse', values='demand_units',
                                 aggfunc='sum', margins=True, margins_name='Total')
            st.table(piv)

            ext_df = df.loc[df['sku_id'] == sku_sel, ['date', 'warehouse', 'demand_units']]
            fig = px.line(ext_df, x='date', y='demand_units', color='warehouse', markers=True,
                          color_discrete_map={'WH_A': "#238CA1", 'WH_B': "#D9916F"},
                          title=f"Demand Units for {sku_sel} by Warehouse")
            fig.update_layout(
                title_font_color="#B14119",
                title_font_size=18
            )
            st.plotly_chart(fig, width='stretch')
            st.write(" ")
            left, right = st.columns([5, 3], gap="medium")
            # Filter đồng thời SKU và Warehouse
            filtered_df = df[(df['sku_id'] == sku_sel) & (df['warehouse'] == wh_sel)]

            # Line chart Demand vs Forecast
            with left:
                ext_df_line = filtered_df[['date', 'demand_units', 'demand_forecast_units']]
                ext_df_line = ext_df_line.rename(columns={'demand_units':'Actual Demand',
                                                          'demand_forecast_units':'Forecast Demand'})
                fig1 = px.line(ext_df_line, x='date', y=['Actual Demand','Forecast Demand'], markers=True,
                               color_discrete_map={'Actual Demand': "#23A140", 'Forecast Demand': "#D9916F"},
                               title=f"Demand vs Forecast for {sku_sel} at {wh_sel}")
                fig1.update_layout(
                    title_font_color="#B14119", title_font_size=18,
                    legend=dict(orientation='h', yanchor='bottom', y=1.08, xanchor='right', x=0.8),
                    margin=dict(t=100)
                )
                st.plotly_chart(fig1, width='stretch')

            # Scatter plot Demand vs Forecast
            with right:
                ext_df_scatter = filtered_df[['demand_units', 'demand_forecast_units']]
                fig2 = px.scatter(ext_df_scatter, x='demand_forecast_units', y='demand_units',
                                  trendline="ols",
                                  title=f"Demand vs Forecast Scatter for {sku_sel} at {wh_sel}")
                max_val = max(ext_df_scatter['demand_forecast_units'].max(),
                              ext_df_scatter['demand_units'].max())
                fig2.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val,
                               line=dict(color='Red', dash='dash'))
                fig2.update_layout(xaxis_title="Forecast Demand", yaxis_title="Actual Demand")
                fig2.update_layout(
                    title_font_color="#B14119",
                    title_font_size=18
                )
                st.plotly_chart(fig2, width='stretch')

    l, r = st.columns([1, 2], gap="medium")
    # Filter theo SKU và Warehouse
    filtered_df = df[(df['sku_id'] == sku_sel) & (df['warehouse'] == wh_sel)]

    with l:
        with st.expander("**:red[Forecast Accuracy Metrics]**", expanded=True):
            st.write(f"**:red[Metrics based on {sku_sel} and {wh_sel}]**")
            mae = mean_absolute_error(filtered_df['demand_units'], filtered_df['demand_forecast_units'])
            rmse = root_mean_squared_error(filtered_df['demand_units'], filtered_df['demand_forecast_units'])
            mape = (abs(filtered_df['demand_units'] - filtered_df['demand_forecast_units']) / filtered_df['demand_units']).mean() * 100
            cols = st.columns(3, gap="medium")
            cols[0].metric("MAE", f"{mae:.2f}")
            cols[1].metric("RMSE", f"{rmse:.2f}")
            cols[2].metric("MAPE", f"{mape:.2f}%")
            st.write(" ")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(filtered_df['demand_diff'], bins=20, color="#9c56c4", edgecolor='black')
            ax.set_title(f"Distribution of Demand Difference for {sku_sel} at {wh_sel}", 
                         fontsize=14, color="#256408", fontweight='bold')
            ax.set_xlabel("Demand Difference (Actual - Forecast)", fontsize=11, color="#2119B1")
            ax.set_ylabel("Frequency", fontsize=11, color="#2319B1")
            plt.grid(alpha=0.25)
            st.pyplot(fig)

            col_sel, col_table = st.columns([1, 4], gap="medium")
            with col_sel:
                st.write(" ")
                metric_sel = st.selectbox("Select Metric", ["MAE", "RMSE"])
            with col_table:
                st.write(f"**:red[Top 5 worst {metric_sel} by SKU]**")
                if metric_sel == "MAE":
                    top5_metric_worst = df.groupby('sku_id').apply(lambda x: mean_absolute_error(x['demand_units'], x['demand_forecast_units']), 
                                                            include_groups=False).nlargest(5).reset_index(name='MAE')
                    top5_metric_best = df.groupby('sku_id').apply(lambda x: mean_absolute_error(x['demand_units'], x['demand_forecast_units']), 
                                                            include_groups=False).nsmallest(5).reset_index(name='MAE')
                else:
                    top5_metric_worst = df.groupby('sku_id').apply(lambda x: root_mean_squared_error(x['demand_units'], x['demand_forecast_units']), 
                                                            include_groups=False).nlargest(5).reset_index(name='RMSE')
                    top5_metric_best = df.groupby('sku_id').apply(lambda x: root_mean_squared_error(x['demand_units'], x['demand_forecast_units']), 
                                                            include_groups=False).nsmallest(5).reset_index(name='RMSE')
                st.table(top5_metric_worst.rename(columns={'sku_id':'SKU-ID'}).set_index('SKU-ID').T)

                st.write(f"**:green[Top 5 best {metric_sel} by SKU]**")                
                st.table(top5_metric_best.rename(columns={'sku_id':'SKU-ID'}).set_index('SKU-ID').T)

    with r:
        with st.expander("**:red[Impact of Exogenous Factors]**", expanded=True):
            c1, c2 = st.columns(2, gap="medium")

            # Promo vs Holiday impact (boxplot)
            with c1:
                fig = px.box(filtered_df, x='promo_flag', y='demand_units', height=360,
                            color='promo_flag', color_discrete_map={1: "#23A140", 0: "#D9916F"},
                            title=f"Impact of Promo on Demand ({sku_sel} at {wh_sel})",
                            labels={'promo_flag':'Promo Flag'})
                fig.update_layout(
                    title_font_color="#B14119",
                    title_font_size=18
                )
                st.plotly_chart(fig, width='stretch')

                fig = px.box(filtered_df, x='holiday_flag', y='demand_units', height=360, 
                            color='holiday_flag', color_discrete_map={1: "#23A140", 0: "#D9916F"},
                            title=f"Impact of Holiday on Demand ({sku_sel} at {wh_sel})",
                            labels={'holiday_flag':'Holiday Flag'})
                fig.update_layout(
                    title_font_color="#B14119",
                    title_font_size=18
                )
                st.plotly_chart(fig, width='stretch')

            # Weather vs Price impact (scatter)
            with c2:
                fig = px.scatter(filtered_df, x='weather_index', y='demand_units', height=360, 
                                trendline="ols",
                                title=f"Weather vs Demand ({sku_sel} at {wh_sel})")
                fig.update_layout(
                    title_font_color="#B14119",
                    title_font_size=18
                )
                st.plotly_chart(fig, width='stretch')

                fig = px.scatter(filtered_df, x='price_usd', y='demand_units', height=360,
                                trendline="ols",
                                title=f"Price vs Demand ({sku_sel} at {wh_sel})")
                fig.update_layout(
                    title_font_color="#B14119",
                    title_font_size=18
                )
                st.plotly_chart(fig, width='stretch')


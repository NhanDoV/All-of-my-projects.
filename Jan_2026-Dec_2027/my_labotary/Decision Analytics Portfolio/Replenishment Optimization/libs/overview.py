import streamlit as st

def data_quality(df_null, data_dim, max_miss_rows, n_miss_cols, n_dupl_rows, total_order_qty, total_cost_usd, total_stockout):
    values = [data_dim, max_miss_rows, n_miss_cols, n_dupl_rows, total_order_qty, total_cost_usd, total_stockout]
    labels = ["◫ Data Dimensions", "🕳️ Số cột có missing values", "⚠️ Max missing rows", "🔁 Duplicate Rows", 
                "📦 Total Order Quantity", "💰 Total Cost (USD)", "❌ Total Stockouts"]
    bg_colors = ["#0C276E", "#162E71", "#192A55", "#112355", "#1A326F", "#061848", "#0C276E"]
    val_colors = ["#880DAD", "#9A7623", "#ED684A", "#19843D", "#2F4807", "#325ABD", "#102F80"]
    label_colors = ["#FFFFFF"] * 7
    cols = st.columns(7, gap='small')

    for i in range(7):
        with cols[i]:
            st.markdown(f"""
                <div style="background: radial-gradient(circle, {bg_colors[i]} 10%, white); padding:10px; border-radius:6px; text-align:center;">
                    <span style="color:{val_colors[i]}; font-weight:bold; font-size:29px;
                                font-family: 'Fira Code', 'Consolas', monospace; 
                                text-shadow: -1px -1px 0 #fff, 1px -1px 0 #fff,
                                            -1px  1px 0 #fff, 1px  1px 0 #fff;">
                        {str(values[i])}
                    </span><br>
                    <span style="color:{label_colors[i]}; font-size:16px">{labels[i]}</span>
                </div>
                """, unsafe_allow_html=True)
    st.write(" ")
    _, c1, _, c2, _ = st.columns([1, 4, 1, 2, 1], gap="small")
    with c1:
        st.markdown("""
            <div style="background: radial-gradient(circle, #0C276E 50%, blue); padding:10px; border-radius:6px; text-align:left;">
                <span style="color:#fffff; font-weight:bold; font-size:16px">
                <ul>
                    <li> Dataset có 10,000 rows, 26 columns </li>
                    <li> Chỉ duy nhất cột 'Region' có 5,000 rows bị thiếu (chiếm 50% dữ liệu) </li>
                    <li> Tổng order ~793k units, tổng cost ~$5M, stockout 1,178 lần. </li>
                </ul>
                </span>
            </div>
            """, unsafe_allow_html=True)
    with c2:
        st.markdown("""<span style='color: #19843D; font-weight: bold; font-size: 18px'>
                       Missing values by warehouse and policy <br>
                    </span>""",
                    unsafe_allow_html=True)
        st.write(" ")
        st.table(df_null)

def data_description():
    c1, _, c2, _, c3 = st.columns([4, 0.25, 5, 0.25, 4], gap="small")

    # Định nghĩa style chung cho background và highlight
    block_style = "background: radial-gradient(circle, #0C276E, #0E49E6); padding:10px; border-radius:6px;"
    highlight = "color:#B4C0DE; font-weight:bold;"

    with c1:
        st.markdown(f"""
        <div style="{block_style}">
            <span style="color:#E8B5A9; font-weight:bold; font-size:19px"> Overview </span>
            <ul>
                <li> <span style="{highlight}">Theme:</span> Operational decision policy (min-max vs base-stock vs ML-style) </li>
                <li> <span style="{highlight}">Granularity:</span> 1 row = 1 SKU–warehouse–day </li>
                <li> <span style="{highlight}">Size:</span> 20 SKUs × 2 warehouses × 250 days = 10,000 rows </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.write(" ")
        st.markdown(f"""
        <div style="{block_style}">
            <ul>
                <li> <span style="{highlight}">Time Keys:</span> date, sku_id, warehouse, region  </li>
                <li> <span style="{highlight}">Decision Variables:</span> policy, order_qty_units, review_flag </li>
                <li> <span style="{highlight}">State Variables:</span> on_hand_units, on_order_units, safety_stock_units, lead_time_days  </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.write(" ")
        st.markdown(f"""
        <div style="{block_style}">
            <ul>
                <li> <span style="{highlight}">Demand Factors:</span> demand_units, demand_forecast_units  </li>
                <li> <span style="{highlight}">Exogenous:</span> promo_flag, holiday_flag, price_usd, weather_index </li>
                <li> <span style="{highlight}">Outcomes:</span> stockout, fill_rate, total_cost_usd  </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.write(" ")

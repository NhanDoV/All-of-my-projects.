import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

# BASKET ==========================================================
def basket_view(basket_df):
    with st.expander("**:blue[CONVERSION_RATE & DISTRUBTION]**", expanded=True):
        conversion_rate_def()
        cols = ['basket_icon_click', 'closed_minibasket_click', 'basket_add_list', 'basket_add_detail']
        dist_rows = st.columns(2, gap='medium')        

        for i in range(4):
            col = cols[i]
            with dist_rows[i % 2]:
                df_cnt = basket_df.groupby(['ordered', col]).size().reset_index()
                df_cnt.columns = ['ordered', col, 'count']
                df_cnt['perc_%'] = (df_cnt['count'] / len(basket_df)).apply(lambda x: f"{100*x:.2f} %")
                df_cnt = df_cnt.replace({1: 'Yes', 0: 'No'})
                
                fig = px.bar(df_cnt, y='ordered', color=col, x='count', 
                             color_discrete_map={'Yes': '#1e3a8a', 'No': '#f87171'}, 
                             log_x=True, text='perc_%', barmode='group', height=320)
                
                fig.update_traces(textfont_size = 16, textangle = 0, textfont_weight='bold',
                                  textposition = "outside", cliponaxis = False)
                
                fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
                
                st.plotly_chart(fig, width='stretch')

# ========== supported functions
def conversion_rate_def():
    st.markdown(
        """
        <span style="color:#fbbf24; font-size:16px; font-weight:600; font-family:Consolas">
            DEFINITION - CONVERSION RATE (action):
        </span>
        """,
        unsafe_allow_html=True,
    )

    st.latex(
        r"""
            P(\text{target}=1 \mid \text{action}=\text{True})
            =
            \frac{\#(\text{target}=1,\ \text{action}=\text{True})}
            {\#(\text{action}=\text{True})}
            """
        )

    st.markdown(
        """
        <span style="color:#86efac; font-size:13px; font-family:Consolas">
            → Proportion of users who convert among users who performed the action.
        </span>
        """,
        unsafe_allow_html=True,
    )

def basket_analytic(df):
    # Define action groups
    action_click = (df["basket_icon_click"].eq(1) | df["closed_minibasket_click"].eq(1))

    action_add = (df["basket_add_list"].eq(1) | df["basket_add_detail"].eq(1))

    # Define mutually exclusive groups
    conditions = [
        ~action_click & ~action_add,
        action_click & ~action_add,
        ~action_click & action_add,
        action_click & action_add,
    ]

    labels = ["No basket action", "Click only", "Add only", "Click + Add",]

    df_group = pd.Series("No basket action", index=df.index)

    for condition, label in zip(conditions, labels):
        df_group.loc[condition] = label

    # Calculate P(ordered=Yes | action_group)
    result = df.assign(action_group=df_group).groupby("action_group", observed=False).agg( 
                                                                                    n_users=("action_group", "size"),
                                                                                    Total_orders=("ordered", 
                                                                                                   lambda x: x.eq(1).sum())).reset_index()

    result["conversion_rate"] = (result["Total_orders"] / result["n_users"]).apply(lambda x: f"{100*x:.2f} %")
    res = result[["action_group", "n_users", "Total_orders", "conversion_rate"]]
    res['n_users'] = res['n_users'].apply(lambda x: f"{x:,}")
    res['Total_orders'] = res['Total_orders'].apply(lambda x: f"{x:,}")

    return res.astype(str)

def basket_comment(basket_df):

    res_table = basket_analytic(basket_df)
    cols = ['basket_icon_click', 'closed_minibasket_click', 'basket_add_list', 'basket_add_detail']

    dictmap = {}
    for col in cols:
        action_yes = basket_df[col].eq(1)
        total_action_yes = action_yes.sum()
        ordered_yes = basket_df.loc[action_yes, "ordered"].eq(1).sum()
        conversion_rate = ordered_yes / total_action_yes if total_action_yes > 0 else 0
        conversion_rate = f"{conversion_rate * 100:.2f} %"
        dictmap[col] = conversion_rate

    st.markdown(f"""
        <div style="
            background: #0f172a;
            border-radius: 12px;
            padding: 24px 28px;
            color: #e2e8f0;
            font-family: 'Segoe UI', system-ui, sans-serif;
            border: 1px solid #334155;
            line-height: 1.5;
        ">

        <h2 style="color: #38bdf8; margin: 0 0 20px 0; font-size: 22px;">
            📊 Conversion Rate Report
        </h2>

        <!-- Individual Action CR -->
        <div style="margin-bottom: 22px;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                <div style="background: linear-gradient(90deg,#1e3a8a,#2563eb); border-radius: 8px; padding: 14px; text-align: center;">
                    <div style="font-size: 26px; font-weight: 700; color: #fff;"> {dictmap['basket_icon_click']} </div>
                    <div style="font-size: 13px; color: #bfdbfe;">basket_icon_click</div>
                </div>
                <div style="background: linear-gradient(90deg,#1e3a8a,#2563eb); border-radius: 8px; padding: 14px; text-align: center;">
                    <div style="font-size: 26px; font-weight: 700; color: #fff;"> {dictmap['basket_add_detail']} </div>
                    <div style="font-size: 13px; color: #bfdbfe;">basket_add_detail</div>
                </div>
                <div style="background: linear-gradient(90deg,#1e3a8a,#2563eb); border-radius: 8px; padding: 14px; text-align: center;">
                    <div style="font-size: 26px; font-weight: 700; color: #fff;"> {dictmap['closed_minibasket_click']} </div>
                    <div style="font-size: 13px; color: #bfdbfe;">closed_minibasket_click</div>
                </div>
                <div style="background: linear-gradient(90deg,#1e3a8a,#2563eb); border-radius: 8px; padding: 14px; text-align: center;">
                    <div style="font-size: 26px; font-weight: 700; color: #fff;"> {dictmap['basket_add_list']} </div>
                    <div style="font-size: 13px; color: #bfdbfe;">basket_add_list</div>
                </div>
            </div>
        </div>

        <!-- Action Group Table -->
        <div style="margin-bottom: 22px;">
            <h3 style="color: #94a3b8; font-size: 15px; margin: 0 0 12px 0; text-transform: uppercase; letter-spacing: 0.5px;">
                Conversion Rate theo nhóm hành động (Action Group)
            </h3>
            <table style="width:100%; border-collapse: collapse; text-align: center; font-size: 14px;">
                <thead>
                    <tr style="background:#1e293b; color:#38bdf8;">
                        <th style="padding:10px; border-bottom:2px solid #334155;">action_group</th>
                        <th style="padding:10px; border-bottom:2px solid #334155;">n_users</th>
                        <th style="padding:10px; border-bottom:2px solid #334155;">Total_orders</th>
                        <th style="padding:10px; border-bottom:2px solid #334155;">conversion_rate</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="background:#0f172a;">
                        <td style="padding:10px; font-weight:600;">Click + Add</td>
                        <td style="padding:10px;"> {res_table.loc[res_table['action_group'] == 'Click + Add', 'n_users'].values[0]} </td>
                        <td style="padding:10px;"> {res_table.loc[res_table['action_group'] == 'Click + Add', 'Total_orders'].values[0]} </td>
                        <td style="padding:10px; color:#4ade80; font-weight:700; font-size:15px;"> {res_table.loc[res_table['action_group'] == 'Click + Add', 'conversion_rate'].values[0]} </td>
                    </tr>
                    <tr>
                        <td style="padding:10px; font-weight:600;">Add only</td>
                        <td style="padding:10px;"> {res_table.loc[res_table['action_group'] == 'Add only', 'n_users'].values[0]} </td>
                        <td style="padding:10px;"> {res_table.loc[res_table['action_group'] == 'Add only', 'Total_orders'].values[0]} </td>
                        <td style="padding:10px; color:#86efac; font-weight:600;"> {res_table.loc[res_table['action_group'] == 'Add only', 'conversion_rate'].values[0]} </td>
                    </tr>
                    <tr style="background:#0f172a;">
                        <td style="padding:10px; font-weight:600;">Click only</td>
                        <td style="padding:10px;"> {res_table.loc[res_table['action_group'] == 'Click only', 'n_users'].values[0]} </td>
                        <td style="padding:10px;"> {res_table.loc[res_table['action_group'] == 'Click only', 'Total_orders'].values[0]} </td>
                        <td style="padding:10px; color:#86efac; font-weight:600;"> {res_table.loc[res_table['action_group'] == 'Click only', 'conversion_rate'].values[0]} </td>
                    </tr>
                    <tr>
                        <td style="padding:10px; font-weight:600;">No basket action</td>
                        <td style="padding:10px;"> {res_table.loc[res_table['action_group'] == 'No basket action', 'n_users'].values[0]} </td>
                        <td style="padding:10px;"> {res_table.loc[res_table['action_group'] == 'No basket action', 'Total_orders'].values[0]} </td>
                        <td style="padding:10px; color:#f87171; font-weight:700;"> {res_table.loc[res_table['action_group'] == 'No basket action', 'conversion_rate'].values[0]} </td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- Nhận xét -->
        <div style="background:#1e293b; border-left:4px solid #fbbf24; border-radius:0 8px 8px 0; padding:16px 18px;">
            <h3 style="color:#fbbf24; margin:0 0 10px 0; font-size:16px;">🔥 Nhận xét nổi bật</h3>
            <ul style="margin:0; padding-left:18px; font-size:14px;">
                <li style="margin-bottom:6px;">
                    <strong style="color:#4ade80;">Click + Add</strong> là nhóm mạnh nhất với <strong>33.95%</strong> — cao gấp hơn 2 lần so với chỉ Click hoặc chỉ Add.
                </li>
                <li style="margin-bottom:6px;">
                    Trong các hành động đơn lẻ, <strong>basket_icon_click</strong> dẫn đầu (<strong>30.07%</strong>), tiếp theo là <strong>basket_add_detail</strong> (27.47%).
                </li>
                <li style="margin-bottom:6px;">
                    Người dùng <strong>không tương tác gì với basket</strong> gần như không convert (<span style="color:#f87171; font-weight:700;">0.20%</span>).
                </li>
                <li>
                    Kết hợp <strong>Click + Add</strong> mang lại hiệu quả chuyển đổi vượt trội rõ rệt so với các hành động đơn lẻ.
                </li>
            </ul>
        </div>

        </div>
        """, unsafe_allow_html=True)

# ============================= WISHLIST

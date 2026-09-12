import streamlit as st

def get_metric_table(data_dim, n_null_cols, max_null_rows, n_dupl_row, n_ordered, n_icon_click, add_to_basket, viewed_checkout, signed_in, wishlist_add):
    st.write(" ")
    row1 = st.columns([4,3,3,3], gap='small')
    val_r1 = [data_dim, n_null_cols, max_null_rows, n_dupl_row]
    labels = ['◫ Data dimension', '▤ n_cols is missed', '▥ max rows missed', '⧉ n_duplicated_rows']

    bg_colors   = ["#0F2A6D"]          # deep navy – still visible on both bg

    val_colors  = [
        "#7C4DFF",   # 0 – electric purple
        "#FFB300",   # 1 – amber / gold
        "#FF6D00",   # 2 – deep orange
        "#00C853",   # 3 – vivid green
        "#00B0FF",   # 4 – bright cyan-blue
        "#FF4081",   # 5 – hot pink
        "#FFD600",   # 6 – bright yellow
    ]
    val_colors = ["#fffff"] * 7

    label_colors = ["#E0E0E0"]         # soft light gray – works on dark & light

    # Tầng 1
    for i in range(4):
        with row1[i]:
            st.markdown(f"""
                <div style="background: linear-gradient(90deg, {bg_colors[0]}, #3166EA, transparent); padding:10px; border-radius:6px; text-align:center;">
                    <span style="color:{val_colors[i]}; font-weight:bold; font-size:29px;
                                font-family: 'Fira Code', 'Consolas', monospace; ">
                        {str(val_r1[i])}
                    </span><br>
                    <span style="color:{label_colors[0]}; font-size:16px">{labels[i]}</span>
                </div>
                """, unsafe_allow_html=True)
    st.write(" ")
    st.write(" ")
    # Tầng 2
    val_r2 = [n_ordered, n_icon_click, add_to_basket]
    labels_2 = ['📦 n_ordered', '🖱️ n_clicks', '🛒 n_add to basket']
    row2 = st.columns(3, gap='small')
    for i in range(3):
        with row2[i]:
            st.markdown(f"""
                <div style="background: linear-gradient(90deg, {bg_colors[0]}, #3166EA, transparent); padding:10px; border-radius:6px; text-align:center;">
                    <span style="color:{val_colors[4+i]}; font-weight:bold; font-size:29px;
                                font-family: 'Fira Code', 'Consolas', monospace; ">
                        {str(val_r2[i])}
                    </span><br>
                    <span style="color:{label_colors[0]}; font-size:16px">{labels_2[i]}</span>
                </div>
                """, unsafe_allow_html=True)
    st.write(" ")
    st.write(" ")
    # Tầng 3
    val_r3 = [viewed_checkout, signed_in, wishlist_add]
    labels_3 = ['👀 n_viewed_checkout', '👥 DISTINCT(user.id)', '❤️ n_added_wishlist']
    row3 = st.columns(3, gap='small')
    for i in range(3):
        with row3[i]:
            st.markdown(f"""
                <div style="background: linear-gradient(90deg, {bg_colors[0]}, #3166EA, transparent); padding:10px; border-radius:6px; text-align:center;">
                    <span style="color:{val_colors[4+i]}; font-weight:bold; font-size:29px;
                                font-family: 'Fira Code', 'Consolas', monospace;">
                        {str(val_r3[i])}
                    </span><br>
                    <span style="color:{label_colors[0]}; font-size:16px">{labels_3[i]}</span>
                </div>
                """, unsafe_allow_html=True)

def abstract_descr(color1, color2, color3):
    html_table = f"""
        <table style="width:100%; border-collapse: collapse; font-family: sans-serif;">
        <thead>
            <tr style="background: linear-gradient(135deg, {color1}, #1e293b 50%, {color1}) ; color: white; text-align:center;">
                <th style="border: 1px solid #ddd; padding: 10px;">Feature group</th>
                <th style="border: 1px solid #ddd; padding: 10px;">Columns</th>
                <th style="border: 1px solid #ddd; padding: 10px;">Meaning</th>
            </tr>
        </thead>
        <tbody>
            <tr style="background: linear-gradient(135deg, {color2}, #A280FF 45%, {color2});">
                <td style="border: 1px solid #ddd; padding: 10px; font-size: 16px; "><b>Tương tác giỏ hàng</b></td>
                <td style="border: 1px solid #ddd; padding: 10px; font-size: 15px;">basket_icon_click, basket_add_list, basket_add_detail, closed_minibasket_click</td>
                <td style="border: 1px solid #ddd; padding: 10px; font-size: 15px;">Người dùng có quan tâm và thêm sản phẩm vào giỏ</td>
            </tr>
            <tr style="background: linear-gradient(135deg, {color3}, #A280FF 45%, {color3});">
                <td style="border: 1px solid #ddd; padding: 10px;"><b>Wishlist & chi tiết sản phẩm</b></td>
                <td style="border: 1px solid #ddd; padding: 10px; font-size: 15px">detail_wishlist_add, list_size_dropdown, image_picker</td>
                <td style="border: 1px solid #ddd; padding: 10px; font-size: 15px">Mức độ quan tâm sâu đến sản phẩm</td>
            </tr>
            <tr style="background: linear-gradient(135deg, {color2}, #A280FF 45%, {color2});">
                <td style="border: 1px solid #ddd; padding: 10px;"><b>Funnel mua hàng</b></td>
                <td style="border: 1px solid #ddd; padding: 10px; font-size: 15px">saw_checkout, checked_delivery_detail, checked_returns_detail, sign_in</td>
                <td style="border: 1px solid #ddd; padding: 10px; font-size: 15px">Tiến gần đến bước thanh toán</td>
            </tr>
            <tr style="background: linear-gradient(135deg, {color3}, #A280FF 45%, {color3});">
                <td style="border: 1px solid #ddd; padding: 10px;"><b>Context</b></td>
                <td style="border: 1px solid #ddd; padding: 10px; font-size: 15px">returning_user, device_computer/tablet, loc_uk, saw_homepage, promo_banner_click</td>
                <td style="border: 1px solid #ddd; padding: 10px; font-size: 15px">Thông tin ngữ cảnh hỗ trợ dự đoán</td>
            </tr>
            <tr style="background:  linear-gradient(135deg, #c84040, #F89F98 90%, #c84040);">
                <td style="border: 1px solid #ddd; padding: 10px;"><b>Target</b></td>
                <td style="border: 1px solid #ddd; padding: 10px; font-size: 15px"> ordered </td>
                <td style="border: 1px solid #ddd; padding: 10px; font-size: 15px">Nhãn dự đoán khả năng mua</td>
            </tr>
        </tbody>
    </table>
    """

    st.markdown(html_table, unsafe_allow_html=True)

def comment():
    with st.expander('**:blue[COMMENT]**', expanded=True):
        st.markdown(
            """
                - Biến target 
                        (<span style="color: #f87171; font-weight: 700;"> ordered </span>) bị 
                        <span style="color: #f87171; font-weight: 700;"> imbalanced </span> 
                cực mạnh, cần thận trọng
                - Ngoài <span style="color: #FFD600; font-weight: 700;"> userid </span>, 
                        tất cả đều là các biến <span style="color: #f87171; font-weight: 700;"> nhị phân </span>
                - Data này là dạng <span style="color: #00B0FF; font-weight: 700;"> aggregated user-level </span> 
                    (mỗi user 1 dòng, các cột là flag đã từng làm hành động đó), không phải session-level hay sequence. 
                    Vì vậy, model sẽ học <span style="color: #A280FF; font-weight: 700;"> “user nào có hành vi X thì propensity cao” </span>, chứ không học được thứ tự hành động (sequence).
            """, unsafe_allow_html=True
        )
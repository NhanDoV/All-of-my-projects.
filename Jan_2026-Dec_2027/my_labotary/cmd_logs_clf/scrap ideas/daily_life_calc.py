import streamlit as st
import pandas as pd    # dùng để in ra bảng biểu suất / phí

with st.sidebar:
    topic = st.radio('**:red[Category]**', ['Điện', 'Thuế', 'Lãi suất'])

st.title(':rainbow[Daily Life Calculation]')
st.set_page_config(page_title="Daily Life Calculation", page_icon="⚡", layout="wide")

# ================= Functions ==============================
def preparing_cost_df(df, x_cal):
    # compute, cân nhắc có nên tạo df.copy hay k?
    df['included_range'] = df.apply(lambda x: (x['val_upto'] < x_cal) or \
                                              ( (x['val_upto'] > x_cal) and (x['val_prev'] <= x_cal) )
                                    ,axis = 1)
    return df

def mapping_calc(x, x_cal):
    if x['included_range'] == False:
        return 0
    else:
        if x['val_upto'] <= x_cal:
            return x['val_upto'] * x['x.base']
        else:
            return (x_cal - x['val_prev']) * x['x.base']

def agg_cost_df(df, x_cal):
    # cân nhắc có nên tạo bản sao df.copy hay k
    df['included_amount'] = df.apply(lambda x: mapping_calc(x, x_cal), axis=1)
    # Lúc này có thể tính tổng ở dãy included_amount để show kq sau cùng; cũng như row-by-row để lấy chi tiết từng bậc thang

    return df

if topic == 'Điện':
    c1, c2 = st.columns([1, 3], gap='large')
    with c1:
        etype = st.radio('Loại hình', ['Dân dụng', 'Hộ kinh doanh'])

        # nếu user tự nhập, show tiếp các option về số bậc khung giá điện và mức tối thiểu / tối đa; otherwise tạo dict có sẵn
        # sẽ update sau khi làm MathCheater xong
        cgfee = st.selectbox('Biểu phí', ['Mặc định', 'Người dùng tự nhập'])

elif topic == "Thuế":
    c1, c2 = st.columns([1, 3], gap='large')
    with c1:
        n_dep = st.number_input('Số người phụ thuộc', )
        t_sal = st.number_input('Tổng lương')
        b_sal = st.number_input('Lương cơ bản', help='dùng để tính BHXH nếu bạn làm các cty như F')
        # nếu user tự nhập, show tiếp các option về số bậc thuế TNCN và mức tối thiểu / tối đa; otherwise tạo dict có sẵn
        txfee = st.selectbox('Biểu suất thuế', ['Mặc định', 'Người dùng tự nhập'])

elif topic == "Lãi suất":
    c1, c2 = st.columns([1, 3], gap='large')
    with c1:
        itype = st.radio('Mô hình lãi', ['Lãi đơn', 'lãi kép'])
        i_dur = st.radio('Kỳ hạn theo', ['ngày', 'tuần', 'tháng', 'năm'])
        durat = st.number_input('Chọn độ dài kỳ hạn')
        irate = st.number_input('Mức lãi suất')
        abase = st.number_input('Số vốn ban đầu')
import streamlit as st
import time

st.set_page_config(page_title="Liện Chat", layout="wide")
st.title("🐶Chat🐶Bot🐶")

fun_text_1 = """
            Ngày xửa ngày xưa, có một ngôi làng tên là **Làng Liện**. Dân làng nơi đây có một phong tục kỳ lạ: mỗi khi đến lễ hội mùa, họ lại ném, quăng, chọi và… liện nào là cà chua, củ cải thối, thậm chí cả trứng gà ung vào trưởng làng và các pháp sư. Ấy thế mà chẳng ai buồn giận, ngược lại còn cười vang cả cánh đồng. Đó chính là cách người dân bày tỏ lòng biết ơn với pháp sư MONKEY – vị pháp sư năm xưa đã từ chối dựng tượng cho gia đình mình để dồn hết tài nguyên phát triển ruộng đồng, nhà cửa và đường sá của dân làng.

            Một ngày nọ, trời sáng như bạc, trăng vẫn chưa chịu lặn, thì từ phương xa xuất hiện một đoàn khách lạ: tộc **Dã Nhân Mặt Trăng**. Họ lạ lắm: tóc bạc như sương, da trắng như khói, mắt lúc nào cũng sáng xanh như ánh trăng rằm. Tộc ấy cử một gã thất phu tên là **`ĐỈNVI`** xuống làng, theo sau hắn là một chú chó trông dữ tợn mà dân làng chỉ gọi bằng đúng một tên: **ĐỰC**.

            Lúc đầu, dân làng Liện thấy lạ lẫm, e dè. Nhưng rồi, chú chó ĐỰC lại chẳng hề hung hãn. Nó lanh lợi, đôi khi còn biết… lắc đầu nguẩy đuôi trước những quả cà chua bay đến. Dân làng dần yêu quý nó, coi như con vật đem may mắn từ mặt trăng về.

            Thời gian trôi đi, trưởng làng Liện – vị lão hiền đức từng chịu bao lần “liện” cà chua, đã qua đời vì tuổi già và bạo bệnh. Tang lễ kéo dài suốt ba tuần, người dân khóc thương ông, củ cải thối cũng chẳng ai buồn quăng. Nhưng trong nỗi đau ấy, một cơ hội ngấm ngầm xuất hiện: ThomasLon, kẻ đối thủ truyền kiếp của vị trưởng làng quá cố, vùng lên và giành lấy chức trưởng thôn.

            Khi lên nắm quyền, **ThomasLon** liền tiến cử gã thất phu ĐỈNVI làm cán bộ phụ trách cầu đường. Nghe có vẻ hợp lý, nhưng dân làng biết rõ hắn chỉ giỏi ăn nhậu chứ chẳng biết tính toán. Thế nhưng, chuyện kỳ lạ hơn nằm ở chú chó ĐỰC. ThomasLon giao cho nó một nhiệm vụ chưa từng có trong lịch sử: “Mày hãy đục mái nhà của thôn xóm để mọi người có thêm ánh sáng!”

            Thoạt nghe, ai cũng bật cười: chó thì biết gì mà đục mái nhà? Nhưng ĐỰC lại có tài lạ. Nó không dùng răng, cũng chẳng dùng vuốt, mà mỗi đêm, dưới ánh trăng, đôi mắt nó lóe sáng, soi vào mái nhà nào là mái nhà ấy… tự nứt ra một ô cửa sổ. Người ta bắt đầu có thêm ánh sáng vào buổi trưa oi bức, buổi chiều mát lành, và buổi tối thì đón cả ánh trăng ùa vào.

            Câu chuyện về chú chó ĐỰC lan đi khắp nơi. Người ta kể lại rằng, nếu có một đêm trăng sáng, bạn ngước nhìn lên mái nhà cũ kỹ mà thấy vệt sáng hình con chó, thì đó chính là ĐỰC – chú chó của làng Liện – đang mải miết rong chơi, mở thêm cửa trời cho con người đón ánh sáng.

            Và thế là, từ ngôi làng luôn liện củ cải và trứng thối, lại nảy sinh một huyền thoại lung linh: huyền thoại về chú chó ĐỰC – kẻ đục mái nhà để đem ánh sáng cho muôn dân.
"""
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Input
user_input = st.chat_input("Bro muốn hỏi đéo gì?")
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Xác định reply
    if "chó" in user_input:
        full_reply = fun_text_1
    else:
        full_reply = "Bot: Mình chưa hiểu rõ lắm, bro nói thêm đi?"

    # Thêm slot rỗng để lát nữa fill typing effect
    st.session_state["messages"].append({"role": "bot", "content": ""})

    # Chạy typing effect
    placeholder = st.empty()
    typed = ""
    for char in full_reply:
        typed += char
        placeholder.markdown(typed)
        time.sleep(0.01)  # tốc độ gõ chữ
    # Cập nhật nội dung cuối cùng vào session_state
    st.session_state["messages"][-1]["content"] = full_reply

# Hiển thị toàn bộ hội thoại
for msg in st.session_state["messages"]:
    if msg["content"] != "":  # bỏ slot rỗng
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"{msg['content']}")
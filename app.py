import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import requests
from io import BytesIO
import pandas as pd

# --- 1. Cấu hình trang ---
st.set_page_config(
    page_title="AI Image Classifier Demo", 
    page_icon="🖼️",
    layout="centered"
)

MODEL_PATH = 'model.keras' 

# Theo mặc định của image_dataset_from_directory, các class được tự động sắp xếp theo alphabet.
# 'Cristiano Ronaldo' (C) sẽ là 0, 'Lionel Messi' (L) sẽ là 1.
# Tuy nhiên, hãy điều chỉnh lại thứ tự này nếu dữ liệu của bạn có thư mục khác.
CLASS_NAMES = ['Cristiano Ronaldo', 'Lionel Messi'] 

IMG_SIZE = (128, 128) 

# --- 2. Load model (cache) ---
@st.cache_resource
def load_my_model():
    model = load_model(MODEL_PATH)
    return model

model = load_my_model()

# --- 3. UI ---
st.title("🖼️ Demo AI Nhận Dạng Ảnh (Image Classification)")
st.markdown("""
Ứng dụng này sử dụng mô hình AI đã được huấn luyện để nhận dạng nội dung của một tấm ảnh (Messi hoặc Ronaldo).
Cung cấp một tấm ảnh bằng cách **tải file** hoặc **dán link ảnh** để xem kết quả dự đoán.
""")

st.divider()

# --- 4. Input ---
tab1, tab2 = st.tabs(["📁 Tải ảnh lên", "🔗 Dán Link Ảnh"])

image = None

with tab1:
    uploaded_file = st.file_uploader(
        "Chọn ảnh (jpg, png, jpeg)...", 
        type=["jpg", "png", "jpeg", "webp"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

with tab2:
    url = st.text_input("Dán URL ảnh vào đây:")
    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            st.error(f"❌ Lỗi tải ảnh: {e}")

# --- 5. Predict ---
if image is not None:
    st.image(image, caption='Ảnh đầu vào', use_container_width=True)

    if st.button('Bắt đầu Dự đoán', type="primary"):

        with st.spinner('Đang xử lý...'):
            try:
                # Preprocess: Thay đổi kích thước ảnh
                processed_image = ImageOps.fit(
                    image, IMG_SIZE, Image.Resampling.LANCZOS
                )
                
                # Chuyển đổi sang chuẩn RGB (để phòng trường hợp ảnh PNG có kênh alpha trong suốt)
                processed_image = processed_image.convert('RGB')

                # Chuyển thành array
                img_array = tf.keras.preprocessing.image.img_to_array(processed_image)
                
                # Bỏ bước img_array = img_array / 255.0 vì kiến trúc model của bạn đã có lớp layers.Rescaling(1./255)
                img_batch = np.expand_dims(img_array, axis=0)

                # Predict
                prediction = model.predict(img_batch)
                
                # Model trả về Sigmoid -> Lấy giá trị xác suất từ 0 đến 1
                score = prediction[0][0] 

                # Logic cho Binary Classification
                if score > 0.5:
                    predicted_label = CLASS_NAMES[1]
                    confidence = score * 100
                else:
                    predicted_label = CLASS_NAMES[0]
                    confidence = (1 - score) * 100
                
                # Tạo list xác suất cho biểu đồ
                prob_class_1 = score
                prob_class_0 = 1 - score

                # Output
                st.divider()
                st.subheader("💡 Kết Quả Dự Đoán")

                col1, col2 = st.columns(2)

                with col1:
                    st.success("Mô hình dự đoán:")
                    st.markdown(f"## **{predicted_label}**")

                with col2:
                    st.metric("Confidence", f"{confidence:.2f}%")

                # Chart
                st.divider()
                st.write("Xác suất phân loại:")

                df = pd.DataFrame({
                    "Class": CLASS_NAMES,
                    "Probability": [prob_class_0, prob_class_1]
                })

                st.bar_chart(df.set_index("Class"))

            except Exception as e:
                st.error(f"❌ Lỗi dự đoán: {e}")

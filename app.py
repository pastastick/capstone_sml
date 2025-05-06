import streamlit as st
from PIL import Image
import os

from csad import inference_openvino_modif


# Threshold per kategori
thresholds = {
    "breakfast_box": {"logical": 2.3232, "structural": -1.4716},
    "juice_bottle": {"logical": 13.4014, "structural": 14.5729},
    "pushpins": {"logical": 8.9723, "structural": 3.4282},
    "screw_bag": {"logical": 9.4575, "structural": 4.5171},
    "splicing_connectors": {"logical": 12.4772, "structural": 12.2652}
}

# Mapping nama tampilan ke kategori internal
MODEL_MAP = {
    "Breakfast Box": "breakfast_box",
    "Juice Bottle": "juice_bottle",
    "Pushpins": "pushpins",
    "Screw Bag": "screw_bag",
    "Splicing Connectors": "splicing_connectors"
}

# Fungsi klasifikasi berdasarkan threshold
def classify_anomaly(score, category, logical_threshold, structural_threshold):
    if category =="juice_bottle":
        if score < logical_threshold:
            return "Normal"
        else:
            return "Terdapat Anomaly"
    else:
        if score < structural_threshold:
            return "normal"
        else:
            return "Terdapat Anomaly"

# UI Streamlit
st.title("📷 Deteksi Anomali Gambar")

# Sidebar: Pilih model
add_selectbox = st.sidebar.selectbox(
    "Pilih Model Deteksi",
    options=MODEL_MAP.keys()
)

# Ambil kategori internal berdasarkan pilihan
selected_category = MODEL_MAP[add_selectbox]

# Upload satu file gambar
uploaded_file = st.file_uploader("Upload satu gambar (PNG/JPG)", type=["png", "jpg"])

if uploaded_file is not None:
    # Simpan gambar sementara
    temp_dir = "tempDir"
    os.makedirs(temp_dir, exist_ok=True)
    image_path = os.path.join(temp_dir, uploaded_file.name)

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Tampilkan gambar yang diupload
    image = Image.open(image_path)
    st.image(image, caption="Gambar Diupload", use_container_width=True)

    # Tombol deteksi
    if st.button("Deteksi"):
        with st.spinner("Memproses..."):
            # Jalankan inferensi dengan file tunggal
            score = inference_openvino_modif(image_path, selected_category)

            # Dapatkan threshold
            logical_threshold = thresholds[selected_category]["logical"]
            structural_threshold = thresholds[selected_category]["structural"]

            # Klasifikasikan
            result = classify_anomaly(score, selected_category, logical_threshold, structural_threshold)

            # Tampilkan skor dan hasil
            st.success(f"Anomaly Score: {score:.4f}")
            if result == "Normal":
                st.success(f"✅ Hasil: {result}")
            else:
                st.error(f"⚠️ Hasil: {result}")
            
    # Bersihkan file sementara setelah selesai
    try:
        os.remove(image_path)
    except:
        pass
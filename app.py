import streamlit as st
from configuration import config
from main import AnomalyDetector
import time
from configuration import NeonDatabase
import logging
import matplotlib.dates as mdates
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import asyncio
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title="Anomaly Detection App",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="auto",
)


######## settingan Sidebar ##########################################
# Category Selection Only
category = st.sidebar.selectbox(
    "Choose category", 
    list(config.ALL_CATEGORIES.keys()),
    index=None
)

# Convert display name to internal category name
if category != None:
    category_internal = config.ALL_CATEGORIES[category]

    # Automatically get the appropriate model for this category
    selected_model = config.get_model_for_category(category_internal)

    # Display category selection confirmation
    st.sidebar.badge(
        f"Kamu memilih kategori **{category}**", 
        icon=":material/check:"
    )

    # Display auto-selected model and accuracy
    category_accuracy = config.get_accuracy_for_category(category_internal)
    st.sidebar.info(
        f"Model: **{selected_model}**\n\n"
        f"Model accuracy is **{category_accuracy}%**"
    )

    # Initialize detector with auto-selected model
    ad = AnomalyDetector(category_internal)
    info = ad.get_model_info()


####### settingan body/ bagian tengah ###################################
_, mid, right = st.columns([0.1, 0.6, 0.3])

with mid:
    judul = st.container(border=True)
    st.title("📷 Deteksi Anomali Pada Gambar")
    st.container(border=False, height=10)
    
    uploaded_files = st.file_uploader(
        "Upload Image (PNG/JPG)",
        type=["png", "jpg"],
        accept_multiple_files=True
    )
    st.container(border=False, height=10)
    
    # Process button
    if st.button("Process Images", type="primary"):
        if category == None:
            st.warning("Please select the category first")
        elif not uploaded_files:
            st.warning("Please upload at least one image")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            error_container = st.container()
            
            processed_count = 0
            start_time = time.time()
            
            processed_results = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    # Update progress
                    progress = int((i + 1) / len(uploaded_files) * 100)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    
                    # Get image bytes
                    image_bytes = uploaded_file.getvalue()
                    
                    # Run prediction
                    with st.spinner(f"Analyzing {uploaded_file.name}..."):
                        score, classification = ad.predict(image_bytes)
                        
                    processed_results.append((uploaded_file.name, score, classification))
                    
                    # Save to database
                    with st.spinner(f"Saving results for {uploaded_file.name}..."):
                        success = NeonDatabase.save_image_record(
                            image_name=uploaded_file.name,
                            category=category,  # PARAMETER BARU
                            image_bytes=image_bytes,
                            score=score,
                            classification=classification
                        )
                        
                        if success:
                            processed_count += 1
                            st.toast(f"Saved {uploaded_file.name}", icon="✅")
                        else:
                            error_container.error(f"Failed to save {uploaded_file.name}")
                
                except Exception as e:
                    error_container.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    LOGGER.exception(f"Error processing {uploaded_file.name}")
            
            # Show final status
            if processed_count > 0:
                processing_time = time.time() - start_time
                status_text.success(
                    f"✅ Processed {processed_count}/{len(uploaded_files)} images successfully! "
                    f"Avg time: {processing_time/len(uploaded_files):.2f}s per image"
                )
                
                # Show results for each processed image
                for name, score, classification in processed_results:
                    st.success(f"✅ The '{name}' image is an *{classification}* with a **{score:.3f}** score.")
            progress_bar.empty()
    

## rencana akan aku buat untuk monitor jumlah total input image
with right:
    with st.container(border=True):
        st.subheader("📊 Image Status")
        if uploaded_files:
            st.metric("Total Images", len(uploaded_files))
            st.metric("Selected Category", category)
        else:
            st.info("No images uploaded yet")






kitas, katas = st.columns([0.4, 0.7])

### rencana akan aku buat format FIFO untuk setiap kategori
with kitas:
    output = st.container(border=True)
    output.write("""
             <style>
            .custom-line {
                font-family: monospace;
                white-space: pre;
                color: rgb(60, 110, 208);
                font-size: 24px;
                }
            </style>

            <div class="custom-line"><b> Kategori</b>\t <span style="color:black">30</span></div>
            <div class="custom-line">total\t <span style="color:#22DD22">102</span></div>
             """, unsafe_allow_html=True)
    
    
    
    
    

    
### biarkan kosong dulu 
# Fungsi untuk membuat quality control chart
def create_quality_control_chart():
    # Data dummy: 30 hari bulan November
    dates = pd.date_range(start="2023-11-01", end="2023-11-30")
    np.random.seed(42)
    
    # Generate data persentase anomaly (antara 0-10%)
    anomaly_percent = np.random.uniform(0, 10, 30)
    
    # Buat figure dan axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot garis persentase anomaly
    ax.plot(dates, anomaly_percent, 'o-', color='#1f77b4', linewidth=2, markersize=6, label='Persentase Anomali')
    
    # Hitung rata-rata (center line)
    avg_percent = np.mean(anomaly_percent)
    ax.axhline(y=avg_percent, color='r', linestyle='--', label=f'Rata-rata ({avg_percent:.2f}%)')
    
    # Hitung batas kontrol (3 sigma)
    std_dev = np.std(anomaly_percent)
    upper_limit = avg_percent + 3 * std_dev
    lower_limit = max(0, avg_percent - 3 * std_dev)  # Pastikan tidak negatif
    
    ax.axhline(y=upper_limit, color='g', linestyle=':', label=f'Batas Kontrol Atas ({upper_limit:.2f}%)')
    ax.axhline(y=lower_limit, color='g', linestyle=':', label=f'Batas Kontrol Bawah ({lower_limit:.2f}%)')
    
    # Format tanggal
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    
    # Atur label dan judul
    ax.set_title('Quality Control : P-Chart\nPersentase Anomali Bulan November', fontsize=14)
    ax.set_xlabel('Tanggal', fontsize=12)
    ax.set_ylabel('Persentase Anomali (%)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')
    
    # Atur batas y
    ax.set_ylim(0, max(12, upper_limit + 2))
    
    # Atur layout
    plt.tight_layout()
    return fig

with katas:
    st.markdown("""
    <style>
    .chart-title {
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="chart-title">Quality Control : P-Chart</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; margin-bottom: 20px; color: #7f8c8d;">Persentase Anomali Bulan November</div>', unsafe_allow_html=True)
    
    # Generate dan tampilkan chart
    fig = create_quality_control_chart()
    st.pyplot(fig)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
kiwah, kawah = st.columns([0.5, 0.5])

with kiwah:
    table = st.container(border=True, height=300)
    table.markdown("masih kosong")
    
with kawah:
    tabel = st.container(border=True, height=300)
    tabel.markdown("masih kosong")
















##### settingan foot/ bagian bawah #################################

def convert_image_data(image_data):
    """Convert database image data to bytes for display"""
    if isinstance(image_data, memoryview):
        return bytes(image_data)
    return image_data

with st.expander("Output Gallery"):
    unique_categories = NeonDatabase.get_unique_categories()
    
    # Buat tabs hanya jika ada kategori
    if unique_categories:
        tabs = st.tabs(unique_categories)
    
        for tab, category in zip(tabs, unique_categories):
            with tab:
                # Buat kolom untuk gambar normal dan anomali
                normal_col, anomaly_col = st.columns(2)
                 
                # Gambar Normal
                with normal_col:
                    st.subheader(f"Normal ({category})")
                    normal_images = NeonDatabase.get_images_by_category_and_classification(
                        category, "NORMAL", limit=5
                    )
                    
                    if normal_images:
                        for name, image_data, score in normal_images:
                            st.image(
                                convert_image_data(image_data),
                                caption=f"{name} (Score: {score:.4f})",
                                use_container_width =True
                            )
                    else:
                        st.info(f"No normal images found for {category}")
                        
                # Gambar Anomali
                with anomaly_col:
                    st.subheader(f"Anomaly ({category})")
                    anomaly_images = NeonDatabase.get_images_by_category_and_classification(
                        category, "ANOMALY", limit=5
                    )
                    
                    if anomaly_images:
                        for name, image_data, score in anomaly_images:
                            st.image(
                                convert_image_data(image_data),
                                caption=f"{name} (Score: {score:.4f})",
                                use_container_width =True
                            )
                    else:
                        st.info(f"No anomaly images found for {category}")
                
    else:
        st.info("No images found in database")
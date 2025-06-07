import streamlit as st
from configuration import config
from main import AnomalyDetector
import time
from configuration import NeonDatabase
import logging
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
with katas:
    chart = st.container(border=True, height=200)
    chart.markdown("masih kosong")
    
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
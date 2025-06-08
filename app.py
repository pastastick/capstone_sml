import streamlit as st
from configuration import config
from main import AnomalyDetector
import time
from configuration import NeonDatabase
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta

from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title="Anomaly Detection App",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="auto",
)

# --- Sidebar Configuration ---
# Category Selection
category = st.sidebar.selectbox(
    "Choose Category", 
    list(config.ALL_CATEGORIES.keys()),
    index=None,
    help="Select the category for anomaly detection. The appropriate model will be auto-selected."
)

# Convert display name to internal category name and display model info
if category is not None:
    category_internal = config.ALL_CATEGORIES[category]

    # Automatically get the appropriate model for this category
    selected_model = config.get_model_for_category(category_internal)

    # Display category selection confirmation
    st.sidebar.badge(
        f"Category selected: **{category}**", 
        icon=":material/check:"
    )

    # Display auto-selected model and accuracy
    category_accuracy = config.get_accuracy_for_category(category_internal)
    st.sidebar.info(
        f"**Model:** {selected_model}\n\n"
        f"**Model Accuracy:** {category_accuracy}%"
    )

    # Initialize detector with auto-selected model
    ad = AnomalyDetector(category_internal)

# --- Sample Data for Download ---

zip_path = "sample_data.zip"
with open(zip_path, "rb") as f:
    sample_zip_bytes = BytesIO(f.read())

# Tampilkan tombol download di sidebar
st.sidebar.markdown("---")
st.sidebar.download_button(
    label="📦 Download Sample Data (ZIP)",
    data=sample_zip_bytes,
    file_name="sample_anomaly_detection_data.zip",
    mime="application/zip"
)


# --- Main Content Area ---
_, mid, _,  right = st.columns([0.1, 0.5, 0.05, 0.25])

with mid:
    st.markdown("<h1 style='text-align: left; color: #1b4185;'>📷 Anomaly Detection</h1>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: left; color: #707070;'>Upload images and let our AI detect anomalies.</h6>", unsafe_allow_html=True) # Changed from h3 to h6 for better hierarchy

    st.container(border=False, height=10) # Empty container for spacing
    
    uploaded_files = st.file_uploader(
        "Upload Image(s) (PNG/JPG)",
        type=["png", "jpg"],
        accept_multiple_files=True,
        help="Drag and drop your image files here or click to browse. Multiple files are supported."
    )
    st.container(border=False, height=8) # Empty container for spacing
     
    # Process button
    if st.button("Process Images", type="primary", use_container_width=True):
        if category is None:
            st.warning("Please select a category first in the sidebar.")
        elif not uploaded_files:
            st.warning("Please upload at least one image to process.")
        else:
            # Display a professional-looking progress bar and status
            progress_bar = st.progress(0, text="Processing images...")
            status_text = st.empty()
            error_container = st.container() # Container to display errors clearly
            
            processed_count = 0
            start_time = time.time()
            
            processed_results = [] # To store results for display after processing
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    # Update progress
                    progress = int((i + 1) / len(uploaded_files) * 100)
                    progress_bar.progress(progress, text=f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})...")
                    status_text.text(f"Currently analyzing: {uploaded_file.name}")
                    
                    # Get image bytes
                    image_bytes = uploaded_file.getvalue()
                    
                    # Run prediction
                    with st.spinner(f"Analyzing {uploaded_file.name} using {selected_model} model..."):
                        score, classification = ad.predict(image_bytes)
                        
                    processed_results.append((uploaded_file.name, score, classification))
                    
                    # Save to database
                    with st.spinner(f"Saving results for {uploaded_file.name} to database..."):
                        success = NeonDatabase.save_image_record(
                            image_name=uploaded_file.name,
                            category=category,
                            image_bytes=image_bytes,
                            score=score,
                            classification=classification
                        )
                        
                        if success:
                            processed_count += 1
                            st.toast(f"Result for '{uploaded_file.name}' saved successfully!", icon="✅")
                        else:
                            error_container.error(f"Failed to save results for '{uploaded_file.name}' to database. Please check your connection.")
                
                except Exception as e:
                    error_container.exception(f"❌ An error occurred while processing '{uploaded_file.name}': {str(e)}")
                    LOGGER.exception(f"Error processing {uploaded_file.name}")
            
            # --- Final Processing Summary ---
            progress_bar.empty() # Clear the progress bar after completion
            status_text.empty()  # Clear the status text
            
            if processed_count > 0:
                processing_time = time.time() - start_time
                st.success(
                    f"✅ Successfully processed {processed_count}/{len(uploaded_files)} images! "
                    f"Average processing time: {processing_time/len(uploaded_files):.2f} seconds per image."
                )
                
                # Display individual results in a clean format
                st.subheader("Individual Image Analysis Results:")
                for name, score, classification in processed_results:
                    # Use markdown for styling classification (bold, color)
                    if classification == "ANOMALY":
                        st.markdown(f"- The image '{name}' is classified as **<span style='color:red;'>{classification}</span>** with a score of **{score:.3f}**.", unsafe_allow_html=True)
                    else:
                        st.markdown(f"- The image '{name}' is classified as **<span style='color:green;'>{classification}</span>** with a score of **{score:.3f}**.", unsafe_allow_html=True)
            else:
                st.error("No images were successfully processed. Please check for errors above.")
    
st.markdown("---")


# --- Right Sidebar / Upload Status ---
with right:
    st.subheader("💡 Upload Status")
    with st.container(border=False):
        if uploaded_files:
            st.metric("Total Images", len(uploaded_files))
            st.metric("Selected Category", category if category else "Not selected")
        else:
            st.info("No images uploaded yet.")


# --- Bottom Section for Statistics and Gallery ---

# Function to generate fixed quality control chart data
def generate_fixed_p_chart_data():
    """
    Generates dummy P-Chart data for May 2025 with some points out of control.
    """
    start_date = date(2025, 5, 1)
    end_date = date(2025, 5, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    np.random.seed(42) # for reproducibility

    # Base proportion (average anomaly rate)
    p_bar = 0.05 # 5% average anomaly rate

    # Number of samples (n_i) for each day (dummy value, e.g., 100 images per day)
    # In a real scenario, this would be actual number of items inspected each day
    n_i = 100 
    
    # Generate anomaly proportions with some variability
    # Ensure proportions are between 0 and 1
    anomaly_proportions = np.random.normal(p_bar, 0.015, len(dates)) # Small std dev for general control
    anomaly_proportions = np.clip(anomaly_proportions, 0, 1) # Clip to ensure it's a proportion

    # Introduce some "out-of-control" points manually
    # Example: Day 5 (index 4) has a high anomaly
    if len(anomaly_proportions) > 4:
        anomaly_proportions[4] = 0.08 # High anomaly (8%)
    # Example: Day 20 (index 19) has a low anomaly
    if len(anomaly_proportions) > 19:
        anomaly_proportions[19] = 0.01 # Low anomaly (1%)
    # Example: Day 25 (index 24) has a very high anomaly
    if len(anomaly_proportions) > 24:
        anomaly_proportions[24] = 0.12 # Very high anomaly (12%)

    std_dev_p = np.sqrt(p_bar * (1 - p_bar) / n_i)
    ucl = p_bar + 3 * std_dev_p
    lcl = p_bar - 3 * std_dev_p
    
    # Ensure LCL is not negative
    lcl = max(0, lcl)

    # Create DataFrame for Streamlit chart
    chart_df = pd.DataFrame({
        "Date": dates,
        "Anomaly Proportion": anomaly_proportions,
        "UCL": ucl,
        "LCL": lcl,
        "Average": p_bar
    })
    
    return chart_df

# Generate the fixed P-Chart data
fixed_p_chart_data = generate_fixed_p_chart_data()


kiri, _, kanan = st.columns([0.4, 0.01, 0.5])

with kiri:
    st.subheader("📊 Category Statistics")
    with st.container(border=False):
        category_stats = NeonDatabase.get_category_stats()
        
        if category_stats:
            stats_df = pd.DataFrame(
                category_stats,
                columns=["Category", "Normal", "Anomaly", "Last Updated"]
            )
            
            stats_df = stats_df.sort_values("Last Updated", ascending=False)
            
            st.dataframe(
                stats_df,
                column_config={
                    "Category": st.column_config.TextColumn("Category"),
                    "Normal": st.column_config.NumberColumn("Normal", format="%d"),
                    "Anomaly": st.column_config.NumberColumn("Anomaly", format="%d"),
                    "Last Updated": st.column_config.DatetimeColumn(
                        "Last Updated",
                        format="DD-MM-YYYY HH:mm"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No category statistics available yet. Upload images to generate data.")
    
    
with kanan:
    st.subheader("📈 Anomaly Trend (May 2025 P-Chart)")
    with st.container(border=True):
        
        fixed_p_chart_data_indexed = fixed_p_chart_data.set_index("Date")
        
        st.line_chart(fixed_p_chart_data_indexed[["Anomaly Proportion", "UCL", "LCL", "Average"]],color=["#157bf7","#f48d10","#a80a0a", "#2ec45e"])
        
        st.markdown(
            """
            <h6 style='text-align: left; color: #555;'>
            - <b>Anomaly Proportion (Blue Line):</b> Daily anomaly rate.<br>
            - <b>Average (Orange Line):</b> Overall average anomaly rate.<br>
            - <b>UCL (Green Line):</b> Upper Control Limit.<br>
            - <b>LCL (Red Line):</b> Lower Control Limit.<br>
            <br>
            Points outside the UCL or LCL indicate potential "out-of-control" process variations.
            </h6>
            """, 
            unsafe_allow_html=True
        )

st.container(border=False, height=12)


# --- Footer Section for Output Gallery ---

def convert_image_data(image_data):
    """
    Converts database image data to bytes for Streamlit display.
    Handles different formats like memoryview.
    """
    if isinstance(image_data, memoryview):
        return bytes(image_data)
    return image_data

with st.expander("🖼️ Output Gallery: Recently Processed Images", expanded=True):
    unique_categories = NeonDatabase.get_unique_categories()
    
    # Create tabs only if there are categories in the database
    if unique_categories:
        tabs = st.tabs(unique_categories)
    
        for tab, category in zip(tabs, unique_categories):
            with tab:
                st.markdown(f"<h4 style='color: #FF6347;'>Category: {category}</h4>", unsafe_allow_html=True)
                # Create columns for normal and anomaly images
                normal_col, anomaly_col = st.columns(2)
                 
                # --- Normal Images ---
                with normal_col:
                    st.subheader(f"✅ Normal Images")
                    normal_images = NeonDatabase.get_images_by_category_and_classification(
                        category, "NORMAL", limit=5
                    )
                    
                    if normal_images:
                        for name, image_data, score in normal_images:
                            st.image(
                                convert_image_data(image_data),
                                caption=f"{name} (Score: {score:.4f})",
                                use_container_width=True
                            )
                    else:
                        st.info(f"No 'Normal' images found for the '{category}' category in the database.")
                        
                # --- Anomaly Images ---
                with anomaly_col:
                    st.subheader(f"‼️ Anomaly Images")
                    anomaly_images = NeonDatabase.get_images_by_category_and_classification(
                        category, "ANOMALY", limit=5
                    )
                    
                    if anomaly_images:
                        for name, image_data, score in anomaly_images:
                            st.image(
                                convert_image_data(image_data),
                                caption=f"{name} (Score: {score:.4f})",
                                use_container_width=True
                            )
                    else:
                        st.info(f"No 'Anomaly' images found for the '{category}' category in the database.")
                
    else:
        st.info("No image data found in the database yet. Upload and process images to see the gallery.")

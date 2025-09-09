import streamlit as st
import requests
import pandas as pd

# MCP backend endpoint
BACKEND_URL = "http://localhost:8000"

# Streamlit UI config
st.set_page_config(page_title="AutoML Assistant (MCP)", layout="wide")
st.title("🤖 AutoML Assistant | Powered by MCP Backend")

# Sidebar: CSV uploader
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    st.success("✅ Dataset uploaded successfully!")

    # Read CSV for preview
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    # Send file to backend
    files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
    with st.spinner("🔁 Uploading to MCP..."):
        response = requests.post(f"{BACKEND_URL}/upload_csv", files=files)

    if response.status_code == 200:
        st.success("📦 Dataset successfully loaded into MCP backend.")

        # Target column selection
        st.markdown("### 🎯 Select Target Column for AutoML:")
        target_col = st.selectbox("Select column", df.columns)

        # Run AutoML button
        if st.button("🚀 Run AutoML"):
            with st.spinner("Running AutoML via MCP..."):
                automl_response = requests.post(
                    f"{BACKEND_URL}/run_automl",
                    json={"target_column": target_col}
                )

            if automl_response.status_code == 200:
                result = automl_response.json()
                st.success(f"✅ AutoML completed with model: **{result['model']}**")
                st.metric("Accuracy", f"{result['accuracy']:.4f}")
                st.metric("F1 Score", f"{result['f1_score']:.4f}")
                st.markdown("### 📂 Artifacts:")
                for path in result["artifacts"]:
                    filename = path.split("/")[-1]
                    download_url = f"{BACKEND_URL}/get_artifact?name={filename}"
                    st.markdown(f"- [{filename}]({download_url})")
            else:
                st.error(f"❌ AutoML failed: {automl_response.text}")

    else:
        st.error(f"❌ Failed to upload dataset to MCP: {response.text}")

else:
    st.info("📤 Please upload a CSV file to begin.")

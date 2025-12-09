import streamlit as st
import pandas as pd
import io
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# --- Custom Modules ---
from preprocessor import process_data, create_preprocessor
from models import get_model_dict
from evaluate import evaluate_model

st.set_page_config(page_title="AutoML Platform", page_icon="🤖")
st.title("🤖 AutoML Platform")

# Helper: Clear session state if new file is uploaded
def reset_session_state():
    keys_to_clear = ['df', 'X_processed', 'X_train', 'trained_models', 'training_results']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# ==========================================
# 1. FILE UPLOAD (Cached)
# ==========================================
with st.sidebar:
    st.header("📁 Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'], on_change=reset_session_state)

if not uploaded_file:
    st.info("👈 Please upload a CSV file to get started")
    st.stop()

# Load Data only if not already loaded
if 'df' not in st.session_state:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
else:
    df = st.session_state['df']

# Show Data Preview
with st.expander("📊 Dataset Preview"):
    st.dataframe(df.head(), width='stretch')


# ==========================================
# 2. TASK & TARGET
# ==========================================
st.subheader("🎯 Configuration")
col1, col2 = st.columns(2)

with col1:
    task = st.selectbox(
        "Select Task", 
        ["Classification", "Regression"], 
        key='task', 
        index=None,                 # <--- Start with nothing selected
        placeholder="Choose task..." # <--- Placeholder text
    )

with col2:
    target_col = st.selectbox(
        "Select Target Column", 
        df.columns.tolist(), 
        key='target_col', 
        index=None,                 # <--- Start with nothing selected
        placeholder="Choose target..."
    )

# 🛑 STOP HERE if the user hasn't selected both options yet
if task is None or target_col is None:
    st.warning("Please select a Task and a Target Column to proceed.")
    st.stop() # Stops the script here until selections are made



# ==========================================
# 3. PREPROCESSING (Cached)
# ==========================================
if not target_col:
    st.stop()


st.markdown("---")
st.subheader("🛠️ Preprocessing")

# Check if we need to run preprocessing
if st.button("⚙️ Run Preprocessing"):
    try:
        with st.spinner("Cleaning and processing data..."):
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Identify columns
            numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

            # Process
            X_processed = process_data(X, numerical_cols, categorical_cols)
            
            # Save EVERYTHING to session state
            st.session_state['X_processed'] = X_processed
            st.session_state['y'] = y
            st.session_state['preprocessor'] = create_preprocessor(numerical_cols, categorical_cols)
            
            st.success("✅ Preprocessing complete!")
            
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")

# Display processed data if available
if 'X_processed' in st.session_state:
    with st.expander("🔍 View Processed Data"):
        st.dataframe(st.session_state['X_processed'].head(), width='stretch')
else:
    st.info("Please run preprocessing to proceed.")
    st.stop()



# ==========================================
# 3.5. DATA ANALYSIS (Cached Correlation)
# ==========================================
if 'X_processed' in st.session_state:
    st.markdown("---")
    
    # We use an expander to keep the UI clean
    with st.expander("🔬 Advanced Data Analysis (Correlation Heatmap)"):
        
        # LOGIC: Check if we already have the figure in memory
        if 'corr_fig' in st.session_state:
            st.write("### 📊 Feature Correlation Matrix")
            st.pyplot(st.session_state['corr_fig'])
            
            # Option to refresh/clear if needed
            if st.button("🔄 Recalculate"):
                del st.session_state['corr_fig']
                st.rerun()
                
        else:
            st.info("Correlation matrix is not calculated yet.")
            
            # The calculation only happens when YOU click this button
            if st.button("🔥 Calculate Correlation"):
                with st.spinner("Calculating correlation matrix..."):
                    
                    # 1. Math (Heavy)
                    corr_matrix = st.session_state['X_processed'].corr()
                    
                    # 2. Plotting (Heavy)
                    fig, ax = plt.subplots(figsize=(12, 10))
                    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
                    
                    # 3. Save to Session State
                    st.session_state['corr_fig'] = fig
                    
                    # 4. Rerun to update the UI immediately (hides button, shows chart)
                    st.rerun()


# ==========================================
# 4. TRAIN-TEST SPLIT
# ==========================================
st.markdown("---")
st.subheader("✂️ Train-Test Split")

test_size = st.slider("Test Data Size (%)", 10, 50, 20)

# Check if we need to update the split
# Logic: If X_train missing OR test_size changed, re-split
if 'X_train' not in st.session_state or st.session_state.get('last_test_size') != test_size:
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        st.session_state['X_processed'], 
        st.session_state['y'], 
        test_size=test_size/100, 
        random_state=42
    )
    
    # Store in session state
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test
    st.session_state['last_test_size'] = test_size

# Show split info
c1, c2 = st.columns(2)
c1.info(f"Training: {st.session_state['X_train'].shape[0]} rows")
c2.info(f"Testing: {st.session_state['X_test'].shape[0]} rows")

## ==========================================
# 5. MODEL TRAINING (Live Logs + Persistence)
# ==========================================
st.markdown("---")
st.subheader("🧠 Model Training")

model_dict = get_model_dict(task)

# Empty default to force user selection
selected_models = st.multiselect("Select Algorithms", list(model_dict.keys()), default=[])

if st.button("🚀 Start Training"):
    if not selected_models:
        st.warning("Please select at least one model.")
    else:
        # 1. Reset stored results so we can start fresh
        st.session_state['training_results'] = []
        st.session_state['trained_models'] = {}
        
        # 2. Prepare Data
        X_train = st.session_state['X_train']
        X_test = st.session_state['X_test']
        y_train = st.session_state['y_train']
        y_test = st.session_state['y_test']
        
        # 3. UI Elements
        progress_bar = st.progress(0)
        st.write("### ⏳ Training Logs")
        
        # Create a container so logs appear in a specific area
        log_container = st.container()

        with log_container:
            for i, name in enumerate(selected_models):
                model = model_dict[name]
                
                try:
                    # --- Train & Predict ---
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # --- Evaluate ---
                    mae, rmse, r2 = evaluate_model(y_test, y_pred)
                    r2_perc = r2 * 100
                    
                    # --- LIVE PRINT (The terminal look you wanted) ---
                    st.markdown(f"**{name}**")
                    st.code(f"""Model Training Performance
RMSE: {rmse:.4f}
MAE: {mae:.4f}
R2 score: {r2_perc:.2f}%""")
                    st.markdown("---") # The separator
                    
                    # --- Store Results for later ---
                    st.session_state['training_results'].append({
                        "Model": name, 
                        "MAE": mae, 
                        "RMSE": rmse, 
                        "R2 Score (%)": r2_perc
                    })
                    st.session_state['trained_models'][name] = model
                    
                except Exception as e:
                    st.error(f"❌ Error training {name}: {e}")
                
                # Update progress bar
                progress_bar.progress((i + 1) / len(selected_models))
                time.sleep(0.1) # Tiny pause for better UX
        
        st.success("✅ Training Complete!")

# ==========================================
# 6. RESULTS & DOWNLOAD (Persisted)
# ==========================================
# This runs even if you didn't just click "Train" (e.g., after downloading a file)
if 'training_results' in st.session_state and st.session_state['training_results']:
    
    results_df = pd.DataFrame(st.session_state['training_results'])
    
    # Leaderboard
    st.subheader("🏆 Leaderboard")
    st.dataframe(results_df.sort_values(by="R2 Score (%)", ascending=False), width='stretch')
    
    # Graph
    st.subheader("📊 Visual Comparison")
    st.bar_chart(results_df.set_index("Model")["R2 Score (%)"])
    
    # Downloads
    st.subheader("💾 Downloads Models for future use")
    
    # 1. Preprocessor
    if 'preprocessor' in st.session_state:
        buf = io.BytesIO()
        pickle.dump(st.session_state['preprocessor'], buf)
        buf.seek(0)
        st.download_button("⬇️ Download Preprocessor (scaler.pkl)", buf, "preprocessor.pkl")
    
    # 2. Models (Iterate through the saved dictionary)
    trained_models = st.session_state.get('trained_models', {})
    
    # Display download buttons in a grid
    cols = st.columns(2)
    for i, (name, model) in enumerate(trained_models.items()):
        buf = io.BytesIO()
        pickle.dump(model, buf)
        buf.seek(0)
        file_name = f"{name.replace(' ', '_').lower()}.pkl"
        
        with cols[i % 2]: # Alternate columns
            st.download_button(
                label=f"⬇️ {name}", 
                data=buf, 
                file_name=file_name, 
                key=name
            )
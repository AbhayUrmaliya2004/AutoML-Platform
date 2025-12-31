import streamlit as st
import pandas as pd
import time, io, pickle
from preprocessor import create_preprocessor, process_data
from models import get_model_dict
from evaluate import evaluate_model
from visualize import plot_numerical, plot_categorical, correlation_heatmap
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config("AutoML Platform", "🤖", layout="wide")
st.sidebar.title("🤖 AutoML Platform")

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def reset_all():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

# --------------------------------------------------
# Upload Dataset
# --------------------------------------------------
uploaded = st.sidebar.file_uploader(
    "📁 Upload CSV",
    type=["csv"],
    on_change=reset_all
)

if uploaded:
    st.session_state["raw_df"] = pd.read_csv(uploaded)

if "raw_df" not in st.session_state:
    st.info("👈 Upload a dataset to begin")
    st.stop()

df = st.session_state["raw_df"]

# --------------------------------------------------
# Navigation
# --------------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    ["Explore Data", "Visualize Data", "Preprocess & Compare", "Train Models"]
)

# ==================================================
# 1️⃣ EXPLORE DATA
# ==================================================
if page == "Explore Data":
    st.header("📊 Explore Data")
    st.subheader("Dataset Preview:")
    st.dataframe(df.head(10))

    col = st.selectbox("Select Column to Explore", df.columns)
    fig_key = f"explore_fig_{col}"

    if fig_key not in st.session_state:
        if df[col].dtype in ["int64", "float64"]:
            fig = plot_numerical(col, df, "Raw Data Distribution")
        else:
            fig = plot_categorical(col, df, "Raw Data Distribution")
        st.session_state[fig_key] = fig

    st.pyplot(st.session_state[fig_key])

# ==================================================
# 2️⃣ VISUALIZE DATA
# ==================================================
if page == "Visualize Data":
    st.header("📈 Dataset Visualizations")

    df_to_visualize = st.session_state.get("X_processed", df)

    if "corr_fig" not in st.session_state:
        num_df = df_to_visualize.select_dtypes(include=["number"])
        st.session_state["corr_fig"] = correlation_heatmap(num_df)

    st.pyplot(st.session_state["corr_fig"])

# ==================================================
# 3️⃣ PREPROCESS & COMPARE
# ==================================================
if page == "Preprocess & Compare":
    st.header("🛠️ Preprocessing & Configuration")

    # Task & target selection
    st.subheader("🎯 Task & Target Selection")
    task = st.selectbox("Select Task", ["Classification", "Regression"], index=0)
    target_col = st.selectbox("Select Target Column", df.columns.tolist(), index=0)

    if st.button("Fix Task and Target Column"):
        st.session_state["task"] = task
        st.session_state["target_col"] = target_col

    if "task" not in st.session_state or "target_col" not in st.session_state: 
        st.warning("Please select Task and Target Column to enable preprocessing.")
        st.stop()

    # Preprocessing options
    scale = st.selectbox("Scaling", ["Standard", "MinMax", "None"])
    impute = st.selectbox("Imputation", ["Mean", "Median"])


    if st.button("⚙️ Run Preprocessing"):
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # Apply preprocessing
        preprocessor = create_preprocessor(num_cols, cat_cols, scale, impute)
        X_processed = process_data(X, preprocessor, fit=True)

        # Store in session
        st.session_state["X_processed"] = X_processed
        st.session_state["y"] = y
        st.session_state["preprocessor"] = preprocessor

        st.success("✅ Preprocessing completed")

    # Comparison section
    if "X_processed" in st.session_state:
        st.subheader("Compare Raw vs Processed Data")
        num_cols = df.drop(columns=[target_col]).select_dtypes(include=["number"]).columns.tolist()
        compare_col = st.selectbox("Select Column to Compare", num_cols)

        # ---------------- Raw
        st.subheader("Before Preprocessing (Raw)")
        st.pyplot(plot_numerical(compare_col, df.drop(columns=[target_col]), "Raw"))

        # ---------------- Processed
        st.subheader("After Preprocessing (Processed)")
        processed_series = st.session_state["X_processed"][compare_col]

        fig, ax = plt.subplots()
        sns.histplot(processed_series, kde=True, bins=30, ax=ax)
        ax.set_title(f"Distribution of {compare_col} (Processed)")
        st.pyplot(fig)

# ==================================================
# 4️⃣ MODEL TRAINING
# ==================================================
if page == "Train Models":
    st.header("🧠 Model Training")

    if "X_processed" not in st.session_state:
        st.warning("⚠️ Preprocessing not done yet. Please run preprocessing first!")
        st.stop()

    X = st.session_state["X_processed"]
    y = st.session_state["y"]
    task = st.session_state["task"]
    target_col = st.session_state["target_col"]

    test_size = st.slider("Test Size (%)", 10, 40, 20)

    # Train-test split
    if ("X_train" not in st.session_state or
        st.session_state.get("last_test_size") != test_size):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, random_state=42
        )
        st.session_state.update({
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "last_test_size": test_size
        })

    model_dict = get_model_dict(task)
    selected_models = st.multiselect(
        "Select Algorithms", list(model_dict.keys()), default=[]
    )

    if st.button("🚀 Start Training"):
        if not selected_models:
            st.warning("Select at least one model")
        else:
            st.session_state["training_results"] = []
            st.session_state["trained_models"] = {}

            progress_bar = st.progress(0)
            st.write("### ⏳ Training Logs")
            log_container = st.container()

            with log_container:
                for i, name in enumerate(selected_models):
                    model = model_dict[name]
                    try:
                        st_time = time.time()
                        model.fit(st.session_state["X_train"], st.session_state["y_train"])
                        end_time = time.time() - st_time
                        preds = model.predict(st.session_state["X_test"])
                        metrics = evaluate_model(st.session_state["y_test"], preds, task)

                        st.markdown(f"**{name}**")
                        st.write(f"Training Time: {(end_time):.2f} seconds")
                        st.code(metrics)

                        st.session_state["training_results"].append({"Model": name, **metrics})
                        st.session_state["trained_models"][name] = model
                    except Exception as e:
                        st.error(f"{name} failed: {e}")

                    progress_bar.progress((i + 1) / len(selected_models))
                    time.sleep(0.1)

            st.success("✅ Training Completed")

    # Results & Downloads
    if "training_results" in st.session_state and st.session_state["training_results"]:
        results_df = pd.DataFrame(st.session_state["training_results"])

        st.subheader("🏆 Leaderboard")
        st.dataframe(results_df, width="stretch")

        st.subheader("📊 Comparison")
        st.bar_chart(results_df.set_index("Model"))

        st.subheader("💾 Downloads")
        if "preprocessor" in st.session_state:
            buf = io.BytesIO()
            pickle.dump(st.session_state["preprocessor"], buf)
            buf.seek(0)
            st.download_button("⬇️ Download Preprocessor", buf, "preprocessor.pkl")

        cols = st.columns(2)
        for i, (name, model) in enumerate(st.session_state["trained_models"].items()):
            buf = io.BytesIO()
            pickle.dump(model, buf)
            buf.seek(0)
            with cols[i % 2]:
                st.download_button(f"⬇️ {name}", buf, f"{name.lower().replace(' ', '_')}.pkl")


### Need to add LLM knowledge to understand more about the data

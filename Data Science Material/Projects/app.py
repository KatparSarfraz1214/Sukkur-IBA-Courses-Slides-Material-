import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="wide",
)

# ---------------------------------------------------------
# Custom styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.6rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #666;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }
        .prediction-box {
            padding: 1.2rem;
            border-radius: 12px;
            text-align: center;
            border: 1px solid rgba(128,128,128,0.25);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Load Iris dataset
# ---------------------------------------------------------
@st.cache_data
def load_data():
    iris = load_iris()

    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    df["species"] = iris.target
    df["species_names"] = df["species"].map(
        lambda x: iris.target_names[x]
    )

    return iris, df


# ---------------------------------------------------------
# Train models
# ---------------------------------------------------------
@st.cache_resource
def train_models():
    iris, df = load_data()

    X = df[iris.feature_names]
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            random_state=42,
            max_iter=200
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5
        ),
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(
                y_test,
                y_pred,
                target_names=iris.target_names,
                output_dict=True,
            ),
            "predictions": y_pred,
        }

    return iris, df, X_train, X_test, y_train, y_test, results


iris, df, X_train, X_test, y_train, y_test, results = train_models()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">🌸 Iris Flower Classification</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    "Machine Learning classification using Logistic Regression and "
    "K-Nearest Neighbors"
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Select a section:",
    [
        "Prediction",
        "Dataset",
        "EDA",
        "Model Evaluation",
    ],
)

selected_model = st.sidebar.selectbox(
    "Classification Model",
    ["Logistic Regression", "K-Nearest Neighbors"],
)

# ---------------------------------------------------------
# Prediction page
# ---------------------------------------------------------
if page == "Prediction":

    st.header("🌱 Predict Iris Species")
    st.write(
        "Enter the four flower measurements below and the trained "
        "model will predict the Iris species."
    )

    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.number_input(
            "Sepal Length (cm)",
            min_value=0.0,
            max_value=15.0,
            value=5.1,
            step=0.1,
        )

        sepal_width = st.number_input(
            "Sepal Width (cm)",
            min_value=0.0,
            max_value=10.0,
            value=3.5,
            step=0.1,
        )

    with col2:
        petal_length = st.number_input(
            "Petal Length (cm)",
            min_value=0.0,
            max_value=15.0,
            value=1.4,
            step=0.1,
        )

        petal_width = st.number_input(
            "Petal Width (cm)",
            min_value=0.0,
            max_value=10.0,
            value=0.2,
            step=0.1,
        )

    input_data = np.array(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
        ]]
    )

    if st.button("🔍 Predict Species", type="primary"):

        model = results[selected_model]["model"]

        prediction = model.predict(input_data)[0]
        predicted_species = iris.target_names[prediction]

        st.markdown("---")

        st.subheader("Prediction Result")

        st.markdown(
            f"""
            <div class="prediction-box">
                <h2>🌸 {predicted_species.title()}</h2>
                <p>Predicted using <b>{selected_model}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Probability, when supported
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]

            st.subheader("Prediction Probabilities")

            probability_df = pd.DataFrame(
                {
                    "Species": iris.target_names,
                    "Probability": probabilities,
                }
            )

            probability_df["Probability"] = (
                probability_df["Probability"] * 100
            ).round(2)

            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True,
            )

            st.bar_chart(
                probability_df.set_index("Species")["Probability"]
            )

# ---------------------------------------------------------
# Dataset page
# ---------------------------------------------------------
elif page == "Dataset":

    st.header("📊 Iris Dataset")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Samples", len(df))

    with col2:
        st.metric("Features", len(iris.feature_names))

    with col3:
        st.metric("Classes", len(iris.target_names))

    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Class Distribution")

    class_counts = df["species_names"].value_counts()

    st.bar_chart(class_counts)

    st.subheader("Feature Statistics")
    st.dataframe(
        df[iris.feature_names].describe(),
        use_container_width=True,
    )

# ---------------------------------------------------------
# EDA page
# ---------------------------------------------------------
elif page == "EDA":

    st.header("📈 Exploratory Data Analysis")

    st.subheader("Pairplot")

    fig = sns.pairplot(
        df,
        hue="species_names",
        vars=iris.feature_names,
        palette="viridis",
    )
    fig.fig.suptitle(
        "Iris Features by Species",
        y=1.02,
    )
    st.pyplot(fig.fig)
    plt.close(fig.fig)

    st.subheader("Feature Box Plots")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, feature in zip(
        axes.ravel(),
        iris.feature_names,
    ):
        sns.boxplot(
            data=df,
            x="species_names",
            y=feature,
            hue="species_names",
            palette="viridis",
            legend=False,
            ax=ax,
        )
        ax.set_title(
            f"Box Plot of {feature.replace(' (cm)', '')}"
        )
        ax.set_xlabel("Species")
        ax.set_ylabel(feature)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Missing Values")

    missing_values = df.isnull().sum().to_frame(
        name="Missing Values"
    )
    st.dataframe(
        missing_values,
        use_container_width=True,
    )

# ---------------------------------------------------------
# Model evaluation page
# ---------------------------------------------------------
elif page == "Model Evaluation":

    st.header("🤖 Model Evaluation")

    selected_result = results[selected_model]

    accuracy = selected_result["accuracy"]

    st.metric(
        f"{selected_model} Accuracy",
        f"{accuracy:.2%}",
    )

    st.subheader("Confusion Matrix")

    cm = selected_result["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names,
        ax=ax,
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(f"{selected_model} Confusion Matrix")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classification Report")

    report = pd.DataFrame(
        selected_result["classification_report"]
    ).transpose()

    st.dataframe(
        report.round(4),
        use_container_width=True,
    )

    st.subheader("Model Comparison")

    comparison_df = pd.DataFrame(
        {
            "Model": list(results.keys()),
            "Accuracy": [
                results[name]["accuracy"]
                for name in results
            ],
        }
    )

    comparison_df["Accuracy"] = (
        comparison_df["Accuracy"] * 100
    ).round(2)

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )

    st.bar_chart(
        comparison_df.set_index("Model")["Accuracy"]
    )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption(
    "Iris Flower Classification | Developed by Sarfraz Ali Katpar | "
    "Data Science Project"
)

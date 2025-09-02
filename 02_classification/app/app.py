import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    precision_recall_curve
)
import random
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🎯 Classification Metrics Playground",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .quiz-correct {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .quiz-incorrect {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'quiz_score' not in st.session_state:
    st.session_state.quiz_score = 0
if 'quiz_total' not in st.session_state:
    st.session_state.quiz_total = 0
if 'answered_questions' not in st.session_state:
    st.session_state.answered_questions = set()
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0

def create_sample_diabetes_dataset():
    """Create a sample diabetes dataset similar to Pima Indians"""
    np.random.seed(42)
    n_samples = 768
    
    data = {
        'Pregnancies': np.random.poisson(3, n_samples),
        'Glucose': np.random.normal(120, 30, n_samples),
        'BloodPressure': np.random.normal(70, 12, n_samples),
        'SkinThickness': np.random.normal(20, 8, n_samples),
        'Insulin': np.random.exponential(100, n_samples),
        'BMI': np.random.normal(32, 7, n_samples),
        'DiabetesPedigreeFunction': np.random.exponential(0.5, n_samples),
        'Age': np.random.gamma(2, 15, n_samples)
    }
    
    # Create target variable with some correlation to features
    diabetes_prob = (
        0.1 * (data['Glucose'] > 140) +
        0.05 * (data['BMI'] > 35) +
        0.03 * (data['Age'] > 50) +
        0.02 * np.array(data['Pregnancies']) +
        0.1
    )
    data['Outcome'] = np.random.binomial(1, np.clip(diabetes_prob, 0, 1), n_samples)
    
    df = pd.DataFrame(data)
    # Ensure non-negative values and realistic ranges
    df = df.abs()
    df['Pregnancies'] = df['Pregnancies'].astype(int)
    df['Age'] = np.clip(df['Age'], 18, 80).astype(int)
    
    return df

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Create an interactive confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted 0', 'Predicted 1'],
        y=['Actual 0', 'Actual 1'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400
    )
    
    return fig

def plot_roc_curve(y_true, y_proba, model_name="Model"):
    """Create an interactive ROC curve plot"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {auc:.3f})',
        line=dict(width=3)
    ))
    
    # Random classifier line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier (AUC = 0.5)',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title=f'ROC Curve - {model_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        showlegend=True
    )
    
    return fig

def calculate_metrics_at_threshold(y_true, y_proba, threshold):
    """Calculate metrics at a specific threshold"""
    y_pred = (y_proba >= threshold).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return accuracy, precision, recall, f1

# Sidebar Navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["🏠 Home", "📚 Fundamentals", "📊 Dataset", "🤖 Model Training", "📈 Metrics Playground", "🧠 Quiz", "💼 Interview Prep"]
)

# Main Title
st.title("🎯 Classification Metrics Learning Playground")
st.markdown("*An interactive app to master classification metrics and ace your ML interviews!*")

# HOME PAGE
if page == "🏠 Home":
    st.header("Welcome to Your ML Learning Journey! 🚀")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎓 What You'll Learn:
        - **Dataset handling** and preprocessing
        - **Model training** with different algorithms
        - **Evaluation metrics** in depth
        - **Interactive visualizations**
        - **Interview preparation** with real questions
        
        ### 🛠️ Features:
        - Load built-in datasets or upload your own
        - Train Logistic Regression & Random Forest
        - Interactive threshold tuning
        - Confusion matrices and ROC curves
        - Quiz system with explanations
        """)
    
    with col2:
        st.markdown("""
        ### 📚 Learning Path:
        1. **📊 Dataset**: Explore and preprocess data
        2. **🤖 Model Training**: Train classification models
        3. **📈 Metrics Playground**: Understand evaluation metrics
        4. **🧠 Quiz**: Test your knowledge
        5. **💼 Interview Prep**: Practice interview questions
        
        ### 🎯 Perfect For:
        - ML beginners and intermediate learners
        - Interview preparation
        - Understanding metric trade-offs
        - Hands-on practice with real datasets
        """)
    
    st.success("👈 Use the sidebar to navigate through different sections!")

# FUNDAMENTALS PAGE
elif page == "📚 Fundamentals":
    st.header("📚 Classification Fundamentals")
    st.markdown("*Master the basic concepts before diving into advanced topics!*")
    
    # Table of Contents
    st.markdown("""
    ### 📋 What You'll Learn:
    1. **Confusion Matrix Building Blocks** (TP, TN, FP, FN)
    2. **Classification Metrics Formulas**
    3. **ROC Curve Deep Dive**
    4. **AUC Interpretation**
    5. **Threshold Selection**
    6. **Common Pitfalls and Best Practices**
    """)
    
    # Section 1: Confusion Matrix Building Blocks
    st.subheader("🧱 1. Confusion Matrix Building Blocks")
    
    st.markdown("""
    Every classification metric starts with these four fundamental values from the **Confusion Matrix**:
    """)
    
    # Create a visual confusion matrix
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Interactive confusion matrix
        st.markdown("""
        ### 📊 Interactive Confusion Matrix
        
        |                    | **Predicted Negative** | **Predicted Positive** |
        |--------------------|-----------------------|------------------------|
        | **Actual Negative** | TN (True Negative)    | FP (False Positive)    |
        | **Actual Positive** | FN (False Negative)   | TP (True Positive)     |
        """)
        
        # Sample confusion matrix with user input
        st.markdown("**Try it yourself! Enter some sample values:**")
        
        col_tp, col_fp = st.columns(2)
        with col_tp:
            tp_val = st.number_input("True Positives (TP)", min_value=0, value=85, step=1)
        with col_fp:
            fp_val = st.number_input("False Positives (FP)", min_value=0, value=15, step=1)
            
        col_fn, col_tn = st.columns(2)
        with col_fn:
            fn_val = st.number_input("False Negatives (FN)", min_value=0, value=20, step=1)
        with col_tn:
            tn_val = st.number_input("True Negatives (TN)", min_value=0, value=80, step=1)
        
        # Calculate metrics with user values
        total = tp_val + fp_val + fn_val + tn_val
        if total > 0:
            accuracy_calc = (tp_val + tn_val) / total
            precision_calc = tp_val / (tp_val + fp_val) if (tp_val + fp_val) > 0 else 0
            recall_calc = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0
            specificity_calc = tn_val / (tn_val + fp_val) if (tn_val + fp_val) > 0 else 0
            f1_calc = 2 * (precision_calc * recall_calc) / (precision_calc + recall_calc) if (precision_calc + recall_calc) > 0 else 0
            
            st.markdown("**📊 Calculated Metrics:**")
            met_col1, met_col2, met_col3 = st.columns(3)
            with met_col1:
                st.metric("Accuracy", f"{accuracy_calc:.3f}")
                st.metric("Precision", f"{precision_calc:.3f}")
            with met_col2:
                st.metric("Recall", f"{recall_calc:.3f}")
                st.metric("Specificity", f"{specificity_calc:.3f}")
            with met_col3:
                st.metric("F1-Score", f"{f1_calc:.3f}")
    
    with col2:
        st.markdown("""
        ### 🎯 Definitions:
        
        **🟢 True Positive (TP)**
        - Correctly predicted positive
        - "We said YES, and it was YES"
        
        **🔴 False Positive (FP)**
        - Incorrectly predicted positive
        - "We said YES, but it was NO"
        - Type I Error
        
        **🟡 False Negative (FN)**
        - Incorrectly predicted negative  
        - "We said NO, but it was YES"
        - Type II Error
        
        **🟢 True Negative (TN)**
        - Correctly predicted negative
        - "We said NO, and it was NO"
        """)
    
    # Section 2: Classification Metrics Formulas
    st.subheader("📐 2. Classification Metrics Formulas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Primary Metrics
        
        **Accuracy** = `(TP + TN) / (TP + TN + FP + FN)`
        - *Overall correctness*
        - Good for balanced datasets
        
        **Precision** = `TP / (TP + FP)`
        - *Quality of positive predictions*
        - "Of all positive predictions, how many were correct?"
        
        **Recall (Sensitivity)** = `TP / (TP + FN)`
        - *Coverage of actual positives*
        - "Of all actual positives, how many did we catch?"
        
        **Specificity** = `TN / (TN + FP)`
        - *Coverage of actual negatives*
        - "Of all actual negatives, how many did we correctly identify?"
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Composite Metrics
        
        **F1-Score** = `2 × (Precision × Recall) / (Precision + Recall)`
        - *Harmonic mean of Precision and Recall*
        - Balances both metrics
        
        **F-Beta Score** = `(1 + β²) × (Precision × Recall) / (β² × Precision + Recall)`
        - *Weighted harmonic mean*
        - β > 1: Favors Recall
        - β < 1: Favors Precision
        
        **Matthews Correlation Coefficient (MCC)** = 
        `(TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))`
        - *Correlation between predicted and actual*
        - Range: -1 to +1 (like Pearson correlation)
        """)
    
    # Section 3: ROC Curve Deep Dive
    st.subheader("📈 3. ROC Curve Deep Dive")
    
    st.markdown("""
    **ROC (Receiver Operating Characteristic)** curve is a fundamental tool for binary classification evaluation.
    """)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Create sample ROC curves
        fig = go.Figure()
        
        # Perfect classifier
        fig.add_trace(go.Scatter(
            x=[0, 0, 1], y=[0, 1, 1],
            mode='lines', name='Perfect Classifier',
            line=dict(color='green', width=3)
        ))
        
        # Good classifier
        fpr_good = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        tpr_good = [0, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
        fig.add_trace(go.Scatter(
            x=fpr_good, y=tpr_good,
            mode='lines', name='Good Classifier (AUC ≈ 0.8)',
            line=dict(color='blue', width=3)
        ))
        
        # Random classifier
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines', name='Random Classifier (AUC = 0.5)',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Poor classifier
        fpr_poor = [0, 0.3, 0.6, 0.8, 1.0]
        tpr_poor = [0, 0.1, 0.2, 0.3, 1.0]
        fig.add_trace(go.Scatter(
            x=fpr_poor, y=tpr_poor,
            mode='lines', name='Poor Classifier (AUC ≈ 0.3)',
            line=dict(color='orange', width=3)
        ))
        
        fig.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate (1 - Specificity)',
            yaxis_title='True Positive Rate (Sensitivity/Recall)',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        ### 🎯 ROC Components:
        
        **X-axis: False Positive Rate (FPR)**
        - `FPR = FP / (FP + TN)`
        - `FPR = 1 - Specificity`
        - "How many negatives were wrongly classified as positive?"
        
        **Y-axis: True Positive Rate (TPR)**
        - `TPR = TP / (TP + FN)`
        - `TPR = Sensitivity = Recall`
        - "How many positives were correctly identified?"
        
        ### 🎯 ROC Interpretation:
        
        **🟢 Perfect Classifier:**
        - Goes straight up, then right
        - AUC = 1.0
        
        **🔵 Good Classifier:**
        - Curves toward top-left
        - AUC > 0.7
        
        **🔴 Random Classifier:**
        - Diagonal line
        - AUC = 0.5
        
        **🟠 Poor Classifier:**
        - Below diagonal
        - AUC < 0.5 (worse than random!)
        """)
    
    # Section 4: AUC Interpretation
    st.subheader("🎯 4. AUC (Area Under Curve) Interpretation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 AUC Score Ranges:
        
        | AUC Score | Interpretation | Performance |
        |-----------|---------------|-------------|
        | **0.9 - 1.0** | Excellent | Outstanding |
        | **0.8 - 0.9** | Good | Very Good |
        | **0.7 - 0.8** | Fair | Acceptable |
        | **0.6 - 0.7** | Poor | Needs Improvement |
        | **0.5 - 0.6** | Fail | Barely Better Than Random |
        | **< 0.5** | Very Poor | Worse Than Random |
        
        ### 🎯 What AUC Tells You:
        - **AUC = 0.8** means there's an 80% chance that the model will rank a randomly chosen positive instance higher than a randomly chosen negative instance
        - **Threshold Independent**: AUC considers all possible thresholds
        - **Scale Invariant**: Measures prediction quality regardless of classification threshold
        """)
    
    with col2:
        st.markdown("""
        ### ⚖️ AUC vs Other Metrics:
        
        **🟢 AUC Advantages:**
        - Threshold independent
        - Scale invariant  
        - Good for comparing models
        - Single number summary
        
        **🔴 AUC Limitations:**
        - Can be optimistic for imbalanced data
        - Doesn't account for class distribution
        - May not reflect business costs of FP vs FN
        
        ### 🎯 When to Use AUC:
        ✅ **Good for:**
        - Model comparison
        - Balanced datasets
        - Ranking/probability problems
        
        ❌ **Be careful with:**
        - Highly imbalanced data
        - When FP and FN have very different costs
        - When you need a specific threshold
        """)
    
    # Section 5: Threshold Selection
    st.subheader("🎚️ 5. Threshold Selection Strategies")
    
    st.markdown("""
    The classification threshold determines the boundary between positive and negative predictions. Here's how to choose it:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Threshold Selection Methods:
        
        **1. Default Threshold (0.5)**
        - Simple and commonly used
        - Good for balanced datasets
        - May not be optimal for your specific use case
        
        **2. ROC-based Selection**
        - **Youden's Index**: Maximize (TPR - FPR)
        - **Closest to Perfect**: Minimize distance to (0,1)
        - **Cost-sensitive**: Minimize total cost
        
        **3. Precision-Recall based**
        - **Maximum F1-Score**: Best balance of P and R
        - **Maximum F-beta**: Weighted toward P or R
        - **Precision/Recall threshold**: Meet minimum requirement
        
        **4. Business-driven**
        - **Cost-based**: Minimize financial impact
        - **Resource-based**: Match available resources
        - **Risk-based**: Acceptable risk tolerance
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Real-world Examples:
        
        **🏥 Medical Diagnosis (High Recall)**
        - Threshold: 0.2 (lower)
        - Goal: Don't miss any diseases
        - Accept more false alarms
        
        **🚨 Fraud Detection (High Precision)**
        - Threshold: 0.8 (higher)
        - Goal: Minimize false alerts
        - Accept missing some fraud
        
        **📧 Spam Filtering (Balanced)**
        - Threshold: 0.5 (default)
        - Goal: Balance spam catching vs false positives
        - F1-score optimization
        
        **🎯 Marketing (Resource-based)**
        - Threshold: Based on campaign capacity
        - Goal: Fill available slots with best prospects
        - Top-k selection approach
        """)
    
    # Section 6: Common Pitfalls and Best Practices
    st.subheader("⚠️ 6. Common Pitfalls and Best Practices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚫 Common Pitfalls:
        
        **1. Accuracy Obsession**
        - Don't rely only on accuracy
        - Especially dangerous with imbalanced data
        
        **2. Ignoring Base Rates**
        - High precision with very rare events can be misleading
        - Always consider class distribution
        
        **3. Wrong Metric for the Job**
        - Using AUC for highly imbalanced data
        - Using accuracy when costs differ
        
        **4. Data Leakage**
        - Including future information in features
        - Using test data for threshold selection
        
        **5. Sample Size Issues**
        - Small positive class leads to unreliable metrics
        - Use confidence intervals
        """)
    
    with col2:
        st.markdown("""
        ### ✅ Best Practices:
        
        **1. Know Your Data**
        - Check class distribution
        - Understand the domain
        - Consider data quality issues
        
        **2. Use Multiple Metrics**
        - Never rely on a single metric
        - Consider precision, recall, F1, AUC
        
        **3. Cross-Validation**
        - Use stratified k-fold
        - Report confidence intervals
        - Check metric stability
        
        **4. Business Context**
        - Understand costs of FP vs FN
        - Consider operational constraints
        - Align metrics with business goals
        
        **5. Visualization**
        - Plot confusion matrices
        - Use ROC and PR curves
        - Show threshold effects
        """)
    
    # Quick Reference Card
    st.subheader("🎴 Quick Reference Card")
    
    st.markdown("""
    ### 📋 Formulas Cheat Sheet:
    
    | Metric | Formula | What it Measures | When to Use |
    |--------|---------|------------------|-------------|
    | **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness | Balanced datasets |
    | **Precision** | TP/(TP+FP) | Quality of positive predictions | When FP is costly |
    | **Recall** | TP/(TP+FN) | Coverage of actual positives | When FN is costly |
    | **Specificity** | TN/(TN+FP) | Coverage of actual negatives | When TN matters |
    | **F1-Score** | 2×(P×R)/(P+R) | Balance of P and R | Imbalanced data |
    | **AUC-ROC** | Area under ROC curve | Discrimination ability | Model comparison |
    
    ### 🎯 Memory Aids:
    - **Precision**: "Be precise about positive predictions"
    - **Recall**: "Recall all the actual positives"
    - **F1**: "Find the balance between precision and recall"
    - **AUC**: "All thresholds considered"
    """)
    
    st.success("🎓 Now you have a solid foundation in classification fundamentals!")

# DATASET PAGE
elif page == "📊 Dataset":
    st.header("📊 Dataset Management")
    
    # Dataset source selection
    data_source = st.radio(
        "Choose data source:",
        ["Built-in Diabetes Dataset", "Upload CSV File"]
    )
    
    if data_source == "Built-in Diabetes Dataset":
        with st.spinner("Loading built-in dataset..."):
            df = create_sample_diabetes_dataset()
        st.success("✅ Built-in diabetes dataset loaded!")
        
    else:
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("✅ CSV file uploaded successfully!")
            
            # Let user select target column
            target_col = st.selectbox("Select target column:", df.columns)
            if target_col:
                # Move target to last column
                cols = [col for col in df.columns if col != target_col]
                cols.append(target_col)
                df = df[cols]
                df.rename(columns={target_col: 'Outcome'}, inplace=True)
        else:
            st.info("👆 Please upload a CSV file to continue")
            st.stop()
    
    # Store dataset in session state
    st.session_state.dataset = df
    
    # Dataset Overview
    st.subheader("🔍 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", df.shape[0])
    with col2:
        st.metric("Features", df.shape[1] - 1)
    with col3:
        st.metric("Classes", df['Outcome'].nunique())
    
    # Dataset Preview
    st.subheader("👀 Dataset Preview")
    st.dataframe(df.head(10))
    
    # Class Distribution
    st.subheader("🎯 Class Distribution")
    
    class_counts = df['Outcome'].value_counts()
    class_percentages = df['Outcome'].value_counts(normalize=True) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            values=class_counts.values,
            names=[f'Class {i}' for i in class_counts.index],
            title="Class Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Class Statistics:**")
        for class_val in class_counts.index:
            st.write(f"- Class {class_val}: {class_counts[class_val]} samples ({class_percentages[class_val]:.1f}%)")
        
        if abs(class_percentages.iloc[0] - class_percentages.iloc[1]) > 20:
            st.warning("⚠️ Dataset is imbalanced! This will affect evaluation metrics.")
        else:
            st.success("✅ Dataset is relatively balanced.")
    
    # Train-Test Split Configuration
    st.subheader("✂️ Train-Test Split Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
    with col2:
        use_stratify = st.checkbox("Use stratified sampling", value=True)
    with col3:
        random_state = st.number_input("Random state", 0, 100, 42)
    
    # Perform split
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    if use_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    # Store splits in session state
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    
    # Display split information
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Training Set:**")
        st.write(f"- Size: {len(X_train)} samples")
        train_dist = y_train.value_counts(normalize=True) * 100
        for class_val in train_dist.index:
            st.write(f"- Class {class_val}: {train_dist[class_val]:.1f}%")
    
    with col2:
        st.write("**Test Set:**")
        st.write(f"- Size: {len(X_test)} samples")
        test_dist = y_test.value_counts(normalize=True) * 100
        for class_val in test_dist.index:
            st.write(f"- Class {class_val}: {test_dist[class_val]:.1f}%")
    
    # Feature Scaling Options
    st.subheader("⚖️ Feature Scaling")
    
    scaling_option = st.selectbox(
        "Choose scaling method:",
        ["No Scaling", "Standard Scaler", "MinMax Scaler"]
    )
    
    if scaling_option == "Standard Scaler":
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), 
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), 
            columns=X_test.columns,
            index=X_test.index
        )
        st.session_state.X_train_scaled = X_train_scaled
        st.session_state.X_test_scaled = X_test_scaled
        st.success("✅ Standard scaling applied!")
        
    elif scaling_option == "MinMax Scaler":
        scaler = MinMaxScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), 
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), 
            columns=X_test.columns,
            index=X_test.index
        )
        st.session_state.X_train_scaled = X_train_scaled
        st.session_state.X_test_scaled = X_test_scaled
        st.success("✅ MinMax scaling applied!")
    
    else:
        st.session_state.X_train_scaled = X_train
        st.session_state.X_test_scaled = X_test
        st.info("ℹ️ No scaling applied.")
    
    st.markdown("""
    ### 📚 When to Scale Features:
    - **Standard Scaler**: For algorithms sensitive to scale (Logistic Regression, SVM, Neural Networks)
    - **MinMax Scaler**: When you want features in range [0,1] (Neural Networks, Image processing)
    - **No Scaling**: For tree-based models (Random Forest, Decision Trees, XGBoost)
    """)

# MODEL TRAINING PAGE
elif page == "🤖 Model Training":
    st.header("🤖 Model Training")
    
    # Check if dataset is available
    if 'dataset' not in st.session_state:
        st.error("❌ Please load a dataset first in the Dataset section!")
        st.stop()
    
    # Model selection
    model_type = st.selectbox(
        "Choose a model:",
        ["Logistic Regression", "Random Forest"]
    )
    
    if model_type == "Logistic Regression":
        st.info("📊 **Logistic Regression**: Linear model, requires feature scaling, interpretable coefficients")
        
        # Hyperparameters
        col1, col2 = st.columns(2)
        with col1:
            C = st.slider("Regularization strength (C)", 0.01, 10.0, 1.0, 0.01)
        with col2:
            max_iter = st.slider("Max iterations", 100, 2000, 1000, 100)
        
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        use_scaled = True
        
    else:
        st.info("🌲 **Random Forest**: Tree-based ensemble, doesn't need scaling, provides feature importance")
        
        # Hyperparameters
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("Number of trees", 10, 200, 100, 10)
        with col2:
            max_depth = st.slider("Max depth", 1, 20, 10)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42
        )
        use_scaled = False
    
    # Train model button
    if st.button("🚀 Train Model"):
        with st.spinner("Training model..."):
            # Select appropriate data
            if use_scaled:
                X_train_use = st.session_state.X_train_scaled
                X_test_use = st.session_state.X_test_scaled
            else:
                X_train_use = st.session_state.X_train
                X_test_use = st.session_state.X_test
            
            # Train model
            model.fit(X_train_use, st.session_state.y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train_use)
            y_test_pred = model.predict(X_test_use)
            y_test_proba = model.predict_proba(X_test_use)[:, 1]
            
            # Store results in session state
            st.session_state.model = model
            st.session_state.model_type = model_type
            st.session_state.y_train_pred = y_train_pred
            st.session_state.y_test_pred = y_test_pred
            st.session_state.y_test_proba = y_test_proba
            st.session_state.model_trained = True
            
        st.success("✅ Model trained successfully!")
    
    # Display results if model is trained
    if st.session_state.model_trained:
        st.subheader("📊 Model Performance")
        
        # Calculate metrics
        train_accuracy = accuracy_score(st.session_state.y_train, st.session_state.y_train_pred)
        test_accuracy = accuracy_score(st.session_state.y_test, st.session_state.y_test_pred)
        test_precision = precision_score(st.session_state.y_test, st.session_state.y_test_pred)
        test_recall = recall_score(st.session_state.y_test, st.session_state.y_test_pred)
        test_f1 = f1_score(st.session_state.y_test, st.session_state.y_test_pred)
        test_auc = roc_auc_score(st.session_state.y_test, st.session_state.y_test_proba)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{test_accuracy:.3f}")
        with col2:
            st.metric("Precision", f"{test_precision:.3f}")
        with col3:
            st.metric("Recall", f"{test_recall:.3f}")
        with col4:
            st.metric("F1-Score", f"{test_f1:.3f}")
        
        # Overfitting check
        if abs(train_accuracy - test_accuracy) > 0.1:
            st.warning(f"⚠️ Possible overfitting detected! Training accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}")
        
        # Feature importance (for tree-based models) or coefficients (for linear models)
        if model_type == "Random Forest" and hasattr(st.session_state.model, 'feature_importances_'):
            st.subheader("🎯 Feature Importance")
            
            importance_df = pd.DataFrame({
                'Feature': st.session_state.X_train.columns,
                'Importance': st.session_state.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature', 
                orientation='h',
                title="Feature Importance"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        elif model_type == "Logistic Regression" and hasattr(st.session_state.model, 'coef_'):
            st.subheader("📊 Feature Coefficients")
            
            coef_df = pd.DataFrame({
                'Feature': st.session_state.X_train.columns,
                'Coefficient': st.session_state.model.coef_[0]
            }).sort_values('Coefficient', key=abs, ascending=False)
            
            fig = px.bar(
                coef_df, 
                x='Coefficient', 
                y='Feature', 
                orientation='h',
                title="Feature Coefficients",
                color='Coefficient',
                color_continuous_scale='RdBu'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.success("🎉 Ready to explore metrics in the next section!")

# METRICS PLAYGROUND PAGE
elif page == "📈 Metrics Playground":
    st.header("📈 Metrics Playground")
    
    if not st.session_state.model_trained:
        st.error("❌ Please train a model first in the Model Training section!")
        st.stop()
    
    # Metrics explanations
    st.subheader("📚 Understanding Classification Metrics")
    
    with st.expander("🎯 Accuracy"):
        st.markdown("""
        **Formula:** `(TP + TN) / (TP + TN + FP + FN)`
        
        **What it means:** Percentage of correct predictions
        
        **When to use:** Balanced datasets
        
        **Limitation:** Can be misleading with imbalanced data
        """)
    
    with st.expander("🎯 Precision"):
        st.markdown("""
        **Formula:** `TP / (TP + FP)`
        
        **What it means:** Of all positive predictions, how many were actually positive?
        
        **When to use:** When False Positives are costly (e.g., spam detection, fraud detection)
        
        **Example:** "How precise are our positive predictions?"
        """)
    
    with st.expander("🎯 Recall (Sensitivity)"):
        st.markdown("""
        **Formula:** `TP / (TP + FN)`
        
        **What it means:** Of all actual positives, how many did we correctly identify?
        
        **When to use:** When False Negatives are costly (e.g., medical diagnosis, security screening)
        
        **Example:** "How many actual positives did we catch?"
        """)
    
    with st.expander("🎯 F1-Score"):
        st.markdown("""
        **Formula:** `2 × (Precision × Recall) / (Precision + Recall)`
        
        **What it means:** Harmonic mean of precision and recall
        
        **When to use:** Imbalanced datasets, when you need to balance precision and recall
        
        **Advantage:** Single metric that considers both false positives and false negatives
        """)
    
    # Interactive threshold tuning
    st.subheader("🎚️ Interactive Threshold Tuning")
    st.markdown("Adjust the classification threshold to see how metrics change:")
    
    threshold = st.slider(
        "Classification Threshold", 
        0.0, 1.0, 0.5, 0.01,
        help="Probability threshold for classifying as positive class"
    )
    
    # Calculate metrics at threshold
    accuracy, precision, recall, f1 = calculate_metrics_at_threshold(
        st.session_state.y_test, 
        st.session_state.y_test_proba, 
        threshold
    )
    
    # Display metrics at current threshold
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
    with col2:
        st.metric("Precision", f"{precision:.3f}")
    with col3:
        st.metric("Recall", f"{recall:.3f}")
    with col4:
        st.metric("F1-Score", f"{f1:.3f}")
    
    # Precision-Recall trade-off visualization
    st.subheader("⚖️ Precision-Recall Trade-off")
    
    thresholds = np.arange(0.01, 1.0, 0.01)
    precisions = []
    recalls = []
    
    for t in thresholds:
        _, p, r, _ = calculate_metrics_at_threshold(
            st.session_state.y_test, 
            st.session_state.y_test_proba, 
            t
        )
        precisions.append(p)
        recalls.append(r)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=thresholds, y=precisions,
        mode='lines', name='Precision',
        line=dict(color='blue', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=thresholds, y=recalls,
        mode='lines', name='Recall',
        line=dict(color='red', width=3)
    ))
    
    # Add current threshold line
    fig.add_vline(
        x=threshold, 
        line_dash="dash", 
        line_color="green",
        annotation_text=f"Current: {threshold}"
    )
    
    fig.update_layout(
        title="Precision-Recall vs Threshold",
        xaxis_title="Threshold",
        yaxis_title="Score",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion Matrix
    st.subheader("🔢 Confusion Matrix")
    
    y_pred_threshold = (st.session_state.y_test_proba >= threshold).astype(int)
    cm_fig = plot_confusion_matrix(st.session_state.y_test, y_pred_threshold)
    st.plotly_chart(cm_fig, use_container_width=True)
    
    # ROC Curve
    st.subheader("📈 ROC Curve")
    
    roc_fig = plot_roc_curve(
        st.session_state.y_test, 
        st.session_state.y_test_proba,
        st.session_state.model_type
    )
    st.plotly_chart(roc_fig, use_container_width=True)
    
    # Metrics use cases
    st.subheader("🎯 When to Use Each Metric")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Precision-focused scenarios:**
        - 🚨 Fraud detection (avoid false alarms)
        - 📧 Spam filtering (don't block important emails)
        - 🎯 Targeted advertising (focus on interested users)
        - 🔍 Search engine results (show relevant results)
        """)
    
    with col2:
        st.markdown("""
        **Recall-focused scenarios:**
        - 🏥 Medical diagnosis (don't miss diseases)
        - 🔒 Security screening (catch all threats)
        - 🆘 Emergency detection (don't miss emergencies)
        - 🎣 Customer churn (identify all at-risk customers)
        """)

# QUIZ PAGE
elif page == "🧠 Quiz":
    st.header("🧠 Knowledge Quiz")
    st.markdown("Test your understanding of classification metrics!")
    
    # Quiz questions database (expanded to 20 questions)
    quiz_questions = [
        {
            "id": 1,
            "question": "What's the main difference between Precision and Recall?",
            "options": [
                "Precision focuses on avoiding false positives, Recall on avoiding false negatives",
                "Precision is for balanced datasets, Recall for imbalanced",
                "Precision uses TP+TN, Recall uses only TP",
                "There's no difference, they're the same metric"
            ],
            "correct": 0,
            "explanation": "Precision = TP/(TP+FP) focuses on the accuracy of positive predictions (avoiding false positives). Recall = TP/(TP+FN) focuses on catching all actual positives (avoiding false negatives)."
        },
        {
            "id": 2,
            "question": "Why can Accuracy be misleading for imbalanced datasets?",
            "options": [
                "Accuracy doesn't work with more than 2 classes",
                "A model predicting only the majority class can still have high accuracy",
                "Accuracy requires feature scaling",
                "Accuracy only works with tree-based models"
            ],
            "correct": 1,
            "explanation": "In a dataset with 95% negative and 5% positive samples, a model always predicting 'negative' would have 95% accuracy but would be useless for identifying positive cases."
        },
        {
            "id": 3,
            "question": "When is F1-score more useful than Accuracy?",
            "options": [
                "When you have more than 2 classes",
                "When you want to balance Precision and Recall",
                "When your dataset is perfectly balanced",
                "When you're using tree-based models"
            ],
            "correct": 1,
            "explanation": "F1-score is the harmonic mean of Precision and Recall, making it ideal when you need to balance both metrics, especially in imbalanced datasets."
        },
        {
            "id": 4,
            "question": "What does AUC-ROC measure?",
            "options": [
                "The accuracy at different thresholds",
                "The model's ability to distinguish between classes",
                "The computational cost of the model",
                "The feature importance rankings"
            ],
            "correct": 1,
            "explanation": "AUC-ROC measures the area under the ROC curve, which represents the model's ability to distinguish between classes across all classification thresholds. Higher AUC means better discrimination."
        },
        {
            "id": 5,
            "question": "In medical diagnosis, which metric is typically most important?",
            "options": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score"
            ],
            "correct": 2,
            "explanation": "Recall is crucial in medical diagnosis because missing a disease (false negative) can be life-threatening. It's better to have some false alarms than to miss actual cases."
        },
        {
            "id": 6,
            "question": "What happens to Precision when you lower the classification threshold?",
            "options": [
                "Precision increases",
                "Precision decreases",
                "Precision stays the same",
                "Precision becomes undefined"
            ],
            "correct": 1,
            "explanation": "Lowering the threshold makes the model classify more samples as positive, which typically increases false positives, thus decreasing precision (TP/(TP+FP))."
        },
        {
            "id": 7,
            "question": "Which models require feature scaling?",
            "options": [
                "Random Forest and Decision Trees",
                "Logistic Regression and SVM",
                "XGBoost and LightGBM",
                "All machine learning models"
            ],
            "correct": 1,
            "explanation": "Distance-based and gradient-based algorithms like Logistic Regression, SVM, and Neural Networks require feature scaling. Tree-based models don't need scaling as they make splits based on individual features."
        },
        {
            "id": 8,
            "question": "What is a Type I error in classification?",
            "options": [
                "Missing a positive case (False Negative)",
                "Incorrectly predicting positive (False Positive)",
                "Predicting the wrong probability",
                "Using the wrong algorithm"
            ],
            "correct": 1,
            "explanation": "Type I error is a False Positive - incorrectly rejecting a true null hypothesis. In classification, it means predicting positive when the actual is negative."
        },
        {
            "id": 9,
            "question": "Which metric is threshold-independent?",
            "options": [
                "Accuracy",
                "Precision",
                "Recall",
                "AUC-ROC"
            ],
            "correct": 3,
            "explanation": "AUC-ROC is threshold-independent because it evaluates the model's performance across all possible thresholds, unlike other metrics that depend on a specific threshold."
        },
        {
            "id": 10,
            "question": "What does stratified sampling ensure in train-test split?",
            "options": [
                "Equal sample sizes in train and test",
                "Same class distribution in train and test sets",
                "Random distribution of features",
                "Faster model training"
            ],
            "correct": 1,
            "explanation": "Stratified sampling maintains the same class distribution (proportion of each class) in both training and test sets, ensuring representative splits."
        },
        {
            "id": 11,
            "question": "In fraud detection, which metric should you prioritize?",
            "options": [
                "Accuracy",
                "Precision",
                "Recall",
                "Specificity"
            ],
            "correct": 1,
            "explanation": "In fraud detection, Precision is crucial because false positives (flagging legitimate transactions as fraud) can severely impact customer experience and business operations."
        },
        {
            "id": 12,
            "question": "What is the range of AUC-ROC values?",
            "options": [
                "0 to 1",
                "-1 to 1",
                "0.5 to 1",
                "0 to 100"
            ],
            "correct": 0,
            "explanation": "AUC-ROC ranges from 0 to 1, where 0.5 indicates random performance, 1.0 indicates perfect classification, and values below 0.5 indicate worse than random (but can be inverted)."
        },
        {
            "id": 13,
            "question": "Which metric is most affected by class imbalance?",
            "options": [
                "AUC-ROC",
                "Precision",
                "Accuracy",
                "F1-Score"
            ],
            "correct": 2,
            "explanation": "Accuracy is most affected by class imbalance because it can be artificially high when the model simply predicts the majority class most of the time."
        },
        {
            "id": 14,
            "question": "What does the ROC curve plot?",
            "options": [
                "Precision vs Recall",
                "True Positive Rate vs False Positive Rate",
                "Accuracy vs Threshold",
                "True Positives vs True Negatives"
            ],
            "correct": 1,
            "explanation": "ROC curve plots True Positive Rate (Sensitivity/Recall) on Y-axis vs False Positive Rate (1-Specificity) on X-axis at various threshold settings."
        },
        {
            "id": 15,
            "question": "When would you prefer Precision-Recall curve over ROC curve?",
            "options": [
                "When dataset is balanced",
                "When dataset is highly imbalanced",
                "When using tree-based models",
                "When features are scaled"
            ],
            "correct": 1,
            "explanation": "PR curves are more informative than ROC curves for highly imbalanced datasets because they focus on the positive class performance, which is typically the minority class of interest."
        },
        {
            "id": 16,
            "question": "What is the harmonic mean of Precision and Recall?",
            "options": [
                "Accuracy",
                "Specificity",
                "F1-Score",
                "AUC-ROC"
            ],
            "correct": 2,
            "explanation": "F1-Score is the harmonic mean of Precision and Recall: F1 = 2 × (Precision × Recall) / (Precision + Recall). It provides a single score balancing both metrics."
        },
        {
            "id": 17,
            "question": "Which is NOT a way to handle imbalanced datasets?",
            "options": [
                "SMOTE (Synthetic Minority Oversampling)",
                "Class weighting",
                "Increasing the learning rate",
                "Undersampling majority class"
            ],
            "correct": 2,
            "explanation": "Increasing learning rate is a hyperparameter tuning technique, not a method for handling class imbalance. The other options are all valid imbalance handling techniques."
        },
        {
            "id": 18,
            "question": "What happens to Recall when you raise the classification threshold?",
            "options": [
                "Recall increases",
                "Recall decreases",
                "Recall stays the same",
                "Recall becomes undefined"
            ],
            "correct": 1,
            "explanation": "Raising the threshold makes the model more conservative in predicting positives, which typically decreases the number of true positives caught, thus decreasing recall."
        },
        {
            "id": 19,
            "question": "In a confusion matrix, what does the diagonal represent?",
            "options": [
                "False predictions",
                "True predictions (correct classifications)",
                "Precision values",
                "Recall values"
            ],
            "correct": 1,
            "explanation": "The diagonal of a confusion matrix represents correct predictions - True Positives and True Negatives. Off-diagonal elements represent misclassifications."
        },
        {
            "id": 20,
            "question": "Which metric would be most important for a spam email filter?",
            "options": [
                "Recall (catch all spam)",
                "Precision (avoid blocking important emails)",
                "Accuracy (overall correctness)",
                "Specificity (correctly identify non-spam)"
            ],
            "correct": 1,
            "explanation": "For spam filtering, Precision is crucial because false positives (marking important emails as spam) can cause users to miss critical communications. Some spam getting through is less problematic than blocking legitimate emails."
        }
    ]
    
    # Quiz interface
    total_questions = len(quiz_questions)
    
    # Initialize quiz if not started or reset session state if old format
    if ('quiz_questions_shuffled' not in st.session_state or 
        'current_question' in st.session_state and 'id' not in st.session_state.get('current_question', {})):
        st.session_state.quiz_questions_shuffled = random.sample(quiz_questions, len(quiz_questions))
        st.session_state.answered_questions = set()
        # Clear old format questions
        if 'current_question' in st.session_state:
            del st.session_state.current_question
        if 'quiz_answered' in st.session_state:
            del st.session_state.quiz_answered
    
    # Progress display
    current_progress = len(st.session_state.answered_questions)
    st.progress(current_progress / total_questions)
    st.write(f"Progress: {current_progress}/{total_questions} questions completed")
    
    if st.button("🎲 Get Next Question") and current_progress < total_questions:
        # Find next unanswered question
        available_questions = [q for q in st.session_state.quiz_questions_shuffled 
                             if q['id'] not in st.session_state.answered_questions]
        
        if available_questions:
            st.session_state.current_question = available_questions[0]
            st.session_state.quiz_answered = False
    
    # Reset quiz option
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset Quiz"):
            st.session_state.answered_questions = set()
            st.session_state.quiz_score = 0
            st.session_state.quiz_total = 0
            st.session_state.quiz_questions_shuffled = random.sample(quiz_questions, len(quiz_questions))
            if 'current_question' in st.session_state:
                del st.session_state.current_question
            if 'quiz_answered' in st.session_state:
                del st.session_state.quiz_answered
            st.success("Quiz reset! Click 'Get Next Question' to start.")
    
    with col2:
        if current_progress == total_questions:
            st.success(f"🎉 Quiz Complete! Final Score: {st.session_state.quiz_score}/{total_questions}")
        elif current_progress == 0:
            st.info("👆 Click 'Get Next Question' to start the quiz!")
    
    # Display current question
    if ('current_question' in st.session_state and 
        current_progress < total_questions and 
        'id' in st.session_state.current_question):
        
        question = st.session_state.current_question
        question_number = current_progress + 1
        
        st.subheader(f"❓ Question {question_number}/{total_questions}")
        st.write(f"**{question['question']}**")
        
        if not st.session_state.get('quiz_answered', False):
            user_answer = st.radio(
                "Choose your answer:",
                options=range(len(question['options'])),
                format_func=lambda x: question['options'][x],
                key=f"question_{question['id']}"
            )
            
            if st.button("✅ Submit Answer"):
                st.session_state.quiz_answered = True
                st.session_state.user_answer = user_answer
                st.session_state.quiz_total += 1
                st.session_state.answered_questions.add(question['id'])
                
                if user_answer == question['correct']:
                    st.session_state.quiz_score += 1
                
                st.rerun()
        
        if st.session_state.get('quiz_answered', False):
            correct_answer = question['correct']
            user_answer = st.session_state.user_answer
            
            if user_answer == correct_answer:
                st.markdown('<div class="quiz-correct">✅ <strong>Correct!</strong></div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown('<div class="quiz-incorrect">❌ <strong>Incorrect!</strong></div>', 
                          unsafe_allow_html=True)
                st.write(f"The correct answer was: **{question['options'][correct_answer]}**")
            
            st.info(f"💡 **Explanation:** {question['explanation']}")
            
            # Show running score
            accuracy = (st.session_state.quiz_score / st.session_state.quiz_total) * 100
            st.write(f"📊 Current Score: {st.session_state.quiz_score}/{st.session_state.quiz_total} ({accuracy:.1f}%)")
            
            if st.button("➡️ Next Question") and current_progress < total_questions:
                # Find next question
                available_questions = [q for q in st.session_state.quiz_questions_shuffled 
                                     if q['id'] not in st.session_state.answered_questions]
                
                if available_questions:
                    st.session_state.current_question = available_questions[0]
                    st.session_state.quiz_answered = False
                    st.rerun()
    
    # Final quiz statistics
    if st.session_state.quiz_total > 0:
        st.subheader("📈 Quiz Statistics")
        accuracy = (st.session_state.quiz_score / st.session_state.quiz_total) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Completed", f"{current_progress}/{total_questions}")
        with col2:
            st.metric("Correct", st.session_state.quiz_score)
        with col3:
            st.metric("Accuracy", f"{accuracy:.1f}%")
        with col4:
            if accuracy >= 80:
                st.metric("Grade", "🏆 Excellent")
            elif accuracy >= 70:
                st.metric("Grade", "👍 Good")
            elif accuracy >= 60:
                st.metric("Grade", "📚 Keep Learning")
            else:
                st.metric("Grade", "💪 Practice More")

# INTERVIEW PREP PAGE
elif page == "💼 Interview Prep":
    st.header("💼 Interview Preparation")
    st.markdown("Common ML interview questions about classification metrics with detailed answers")
    
    interview_qa = {
        "🎯 Metrics Fundamentals": {
            "What's the difference between Precision and Recall?": """
            **Precision** = TP/(TP+FP) - "Of all positive predictions, how many were correct?"
            - Focus: Avoiding false positives
            - Use case: Spam detection, fraud detection
            
            **Recall** = TP/(TP+FN) - "Of all actual positives, how many did we catch?"
            - Focus: Avoiding false negatives  
            - Use case: Medical diagnosis, security screening
            
            **Key insight**: There's usually a trade-off between them.
            """,
            
            "Why can Accuracy be misleading?": """
            **Problem**: High accuracy doesn't always mean good model performance.
            
            **Example**: Dataset with 95% negative, 5% positive samples
            - Model always predicting "negative" = 95% accuracy
            - But completely useless for finding positive cases!
            
            **Solution**: Use Precision, Recall, F1-score for imbalanced data.
            """,
            
            "When should you use F1-score vs Accuracy?": """
            **Use F1-score when**:
            - Dataset is imbalanced
            - Both false positives and false negatives matter
            - You need a single metric balancing precision and recall
            
            **Use Accuracy when**:
            - Dataset is balanced
            - All classes are equally important
            - Simple interpretation is needed
            """,
            
            "What does AUC-ROC tell you?": """
            **AUC-ROC** = Area Under ROC Curve
            - Measures model's discrimination ability across all thresholds
            - Range: 0.5 (random) to 1.0 (perfect)
            - Threshold-independent metric
            
            **Interpretation**:
            - 0.9-1.0: Excellent
            - 0.8-0.9: Good
            - 0.7-0.8: Fair
            - 0.5-0.6: Poor
            """
        },
        
        "⚖️ Model Considerations": {
            "When do you need feature scaling?": """
            **Need scaling**:
            - Distance-based: KNN, SVM, K-means
            - Gradient-based: Logistic Regression, Neural Networks
            - Regularized models: Ridge, Lasso
            
            **Don't need scaling**:
            - Tree-based: Random Forest, Decision Trees, XGBoost
            - Naive Bayes
            
            **Why**: Trees make splits on individual features, not distances.
            """,
            
            "How do you handle imbalanced datasets?": """
            **Techniques**:
            1. **Resampling**: SMOTE, undersampling, oversampling
            2. **Class weights**: Penalize minority class mistakes more
            3. **Different metrics**: F1, AUC instead of accuracy
            4. **Ensemble methods**: Random Forest handles imbalance well
            5. **Threshold tuning**: Optimize for business metric
            
            **Key**: Choose technique based on your specific use case.
            """,
            
            "What's the difference between ROC and PR curves?": """
            **ROC Curve** (TPR vs FPR):
            - Good for balanced datasets
            - Less sensitive to class imbalance
            - Standard choice for many applications
            
            **PR Curve** (Precision vs Recall):
            - Better for imbalanced datasets
            - More informative when positive class is rare
            - Shows performance on minority class clearly
            """
        },
        
        "🔄 Process Questions": {
            "Why do we split data into train/test sets?": """
            **Purpose**: Estimate how model will perform on unseen data
            
            **Without split**: Overly optimistic performance estimates
            **With split**: Realistic assessment of generalization
            
            **Best practices**:
            - Use stratified sampling for classification
            - Common splits: 80/20, 70/30
            - Use validation set for hyperparameter tuning
            """,
            
            "What's stratified sampling and why use it?": """
            **Stratified sampling**: Maintains class distribution across splits
            
            **Example**:
            - Original: 70% Class A, 30% Class B
            - Stratified split: Train AND test both have ~70% A, 30% B
            
            **Benefits**:
            - More representative splits
            - Consistent evaluation
            - Especially important for imbalanced data
            """,
            
            "How do you choose the right threshold?": """
            **Methods**:
            1. **Business requirements**: Cost of FP vs FN
            2. **ROC curve**: Point closest to (0,1)
            3. **PR curve**: Best F1-score point
            4. **Cross-validation**: Optimize on validation set
            
            **Example**: Medical diagnosis might use low threshold (high recall) to catch all cases.
            """
        },
        
        "💡 Advanced Topics": {
            "What are Type I and Type II errors?": """
            **Type I Error** (False Positive):
            - Rejecting true null hypothesis
            - "False alarm"
            - Related to Precision (1-Precision shows FP rate)
            
            **Type II Error** (False Negative):
            - Accepting false null hypothesis
            - "Missing the signal"
            - Related to Recall (1-Recall shows FN rate)
            
            **Medical example**:
            - Type I: Healthy person diagnosed as sick
            - Type II: Sick person diagnosed as healthy
            """,
            
            "How do you evaluate multi-class classification?": """
            **Approaches**:
            1. **Macro-average**: Average metrics across classes (treats all classes equally)
            2. **Micro-average**: Global average (influenced by class frequency)
            3. **Weighted average**: Weight by class support
            
            **Additional metrics**:
            - Confusion matrix (NxN for N classes)
            - Per-class precision/recall
            - Cohen's Kappa (accounts for chance agreement)
            """,
            
            "What's cross-validation and why use it?": """
            **Cross-validation**: Multiple train/test splits for robust evaluation
            
            **Types**:
            - **K-fold**: Split into K parts, train on K-1, test on 1
            - **Stratified K-fold**: Maintains class distribution
            - **Leave-One-Out**: Each sample is test set once
            
            **Benefits**:
            - More reliable performance estimates
            - Better use of limited data
            - Helps detect overfitting
            """
        }
    }
    
    # Display Q&A by category
    for category, questions in interview_qa.items():
        st.subheader(category)
        
        for question, answer in questions.items():
            with st.expander(question):
                st.markdown(answer)
    
    # Quick reference card
    st.subheader("🎴 Quick Reference Card")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Key Formulas:**
        - Accuracy = (TP+TN)/(TP+TN+FP+FN)
        - Precision = TP/(TP+FP)
        - Recall = TP/(TP+FN)
        - F1 = 2×(P×R)/(P+R)
        - Specificity = TN/(TN+FP)
        """)
    
    with col2:
        st.markdown("""
        **🎯 When to Use:**
        - **Precision**: Avoid false alarms
        - **Recall**: Don't miss positives
        - **F1**: Balance both, imbalanced data
        - **AUC**: Overall discrimination ability
        - **Accuracy**: Balanced datasets only
        """)
    
    # Tips for interviews
    st.subheader("💡 Interview Tips")
    
    st.markdown("""
    **🎯 How to Answer Metric Questions:**
    1. **Define the metric clearly** with formula
    2. **Explain what it measures** in plain English
    3. **Give a real-world example** of when to use it
    4. **Mention limitations** or trade-offs
    5. **Connect to business impact** when possible
    
    **🗣️ Example Structure:**
    - "Precision is TP/(TP+FP), which measures..."
    - "You'd use this when false positives are costly, like..."
    - "The limitation is that it doesn't account for..."
    - "In practice, you might optimize for this in scenarios where..."
    """)
    
    st.success("🎉 You're ready to ace those ML interviews!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    ❤️ Streamlit App for ML Learning & Interview Prep <br>
    © 2025 Ibrahim Kaiser | With help from AI assistants (ChatGPT & Claude)
</div>
""", unsafe_allow_html=True)
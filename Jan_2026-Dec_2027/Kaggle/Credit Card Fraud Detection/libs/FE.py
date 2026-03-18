import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from imblearn.under_sampling import RandomUnderSampler, TomekLinks      #   !pip install imblearn
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import LabelEncoder

from sklearn.inspection import permutation_importance
import plotly.express as px

import matplotlib.pyplot as plt
import seaborn as sns

bg_color = "#60A5FA"

def styled_df(df):
    
    num_cols = df.select_dtypes(include="number").columns
    styler = df.style.set_properties(**{
                    "background-color": bg_color,  
                    "color": "#1D287A",           
                    "font-weight": "500",
                    "border": "1px solid #BBDEFB", 
                    "padding": "6px",
                }).set_table_styles([
                    {"selector": "th", "props": [
                        ("background-color", "#09052f"),
                        ("color", "white"),
                        ("font-weight", "bold"),
                        ("text-align", "center"),
                    ]}
                ])

    # center numeric columns
    styler = styler.set_properties(
        subset=num_cols,
        **{"text-align": "center"}
    )

    return styler

def show_plotly_template(fig):
    fig.update_layout(
        plot_bgcolor = "rgba(255,255,255,0.15)",
        paper_bgcolor = "rgba(255,255,255,0)",
        font = dict(color = "#f0f2f6"),
        margin = dict(
                        l = 10, # left
                        r = 10, # right
                        t = 40, # top
                        b = 10 # bottom
                    )
    )
    
    return fig

def feature_eningineering_enhancing(df):

    db = df.copy()

    # Night transaction flag
    db["night_transaction"] = db["transaction_hour"].isin([0, 1, 2, 3]).astype(int)

    # High amount flag
    db["high_amount"] = (db["amount"] > 900).astype(int)

    # Low trusted score 
    db['low_scored'] = (db['device_trust_score'] < 39).astype(int)

    return db

def review_new_results(df):    
    notes = {
        'night_transaction': '`transaction_hour` is **in 0am to 4am**',
        'high_amount': 'if `transaction_amount` is **greater than 900**',
        'low_scored': 'if `device_trust_score` is **lower than 39**'
    }
    considered_cols = list(notes.keys())
    with st.expander("New added features", expanded=True):
        cols = st.columns(3)
        for idx, col in enumerate(cols):
            c_col = considered_cols[idx]
            with col:
                with st.expander(c_col, expanded=True):
                    if idx == 0:
                        c1, _, c2 = st.columns([1, 0.02, 1])
                    else:
                        c1, _, c2 = st.columns([4, 0.02, 3])
                    temp = pd.crosstab(df['is_fraud'], df[c_col])
                    with c1:
                        st.write(notes[c_col])
                        st.table(styled_df(temp))
                    with c2:
                        p_AB = ((df['is_fraud'] == "1") & (df[c_col] == 1)).sum()
                        p_B  = (df[c_col] == 1).sum()
                        p_A  = (df['is_fraud'] == "1").sum()

                        p1 = (p_AB/p_B*100) if p_B else 0
                        p2 = (p_AB/p_A*100) if p_A else 0

                        st.metric(f"Prob ( Fraud | {c_col} )", f"{p1:.3f}%", border = True)
                        st.metric(f"Prob ( {c_col} | Fraud )", f"{p2:.3f}%", border = True)

def label_encoding(df):
    le = LabelEncoder()
    df["merchant_category"] = le.fit_transform(df["merchant_category"])

    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    return X, y

def split_and_scaling(X, y):
    feature_names = X.columns.tolist()

    # 1. split first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. scale then
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # fit on train set
    X_test = scaler.transform(X_test)         # transform test

    return X_train, X_test, y_train, y_test, feature_names, scaler

def resampling_preprocessing(df, resampling_name = 'SMOTE'):
    X, y = label_encoding(df)
    if resampling_name == 'SMOTE':
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    elif resampling_name == "TomeLinks":
        tl = TomekLinks()
        X_resampled, y_resampled = tl.fit_resample(X, y)

    return X_resampled, y_resampled

def get_clf_model(model_name, param1, param2):
    # Model selected
    if model_name == "Logistic Regression": # param1, 2 = c_value, penalty
        clf = LogisticRegression(C = param1, l1_ratio = param2, max_iter = 500)
    elif model_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators = param1, max_depth = param2, random_state = 42)
    elif model_name == "Decision Tree":
        clf = DecisionTreeClassifier(max_depth = param1, criterion = param2, random_state = 42)
    elif model_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors = param1, weights = param2)    
    else:
        clf = XGBClassifier(
                    n_estimators=param1,
                    learning_rate=param2,
                    max_depth=10,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric="logloss",
                    n_jobs=-1,
                )
    return clf


def get_importance_features_table(model_name, clf, X_test, y_test, feature_names):
    if model_name == "Logistic Regression":
        perm_importance = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(perm_importance.importances_mean)
        }).sort_values('importance', ascending = True)
    else:
        if hasattr(clf, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,  # ← Dùng feature_names đã lưu
                'importance': clf.feature_importances_
            }).sort_values('importance', ascending=True)
        else:
            perm_importance = permutation_importance(clf, X_test, y_test, n_repeats=10)
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(perm_importance.importances_mean)
            }).sort_values('importance', ascending = True)

    # scaling importance into (0-100)
    importance_df['importance'] = (importance_df['importance'] / importance_df['importance'].sum() * 100).round(5)

    return importance_df

def get_feature_importance_chart(importance_df, key = ""):

    fig_fi = px.bar(importance_df, x='importance', y='feature', 
                    orientation='h', title="Feature Importance (%)",
                    text='importance', color='importance',
                    color_continuous_scale = 'Blues')

    fig_fi.update_layout(
        height = 600,
        coloraxis_colorbar = dict(
            orientation = 'h',
            y = 1.25,
            x = 0.5,
            xanchor = 'center',
            yanchor = 'top',
            len = 0.75,
            thickness = 12,
            tickangle = 0,
            nticks = 4 
        )
    )
    fig_fi = show_plotly_template(fig_fi)
    st.plotly_chart(fig_fi, width = 'content', key = key)

def get_confusion_matrix(y_test, y_pred):

    # Get confusion matrix
    fig, ax = plt.subplots(1, 1, figsize = (5, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot = True, fmt = "d", ax = ax, 
                annot_kws = {"size": 27, "weight": "bold"},
                cmap = "Blues", cbar_kws = {'orientation': 'horizontal', 
                                            'shrink': 0.6, 'pad': -0.05}
                )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

def get_sampler(resampling_method, methodology_name):
    if resampling_method == "Undersampling":
        if methodology_name == "TomeLinks":
            sampler = TomekLinks()
        else:
            sampler = RandomUnderSampler(random_state=42)
    else:
        if methodology_name == "SMOTE":
            sampler = SMOTE(random_state=42)
        else:
            sampler = RandomOverSampler(random_state=42)

    return sampler
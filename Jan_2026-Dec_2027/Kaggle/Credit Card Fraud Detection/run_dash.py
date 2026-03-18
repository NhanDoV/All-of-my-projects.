import streamlit as st
from libs.EDA import *
from libs.FE import *

# Load dataset
df = pd.read_csv("data/credit_card_fraud_10k.csv")

# In numeric-columns, we WILL NOT CONSIDER `transaction_id`
num_cols = [
    "amount", "transaction_hour", "device_trust_score",
    "velocity_last_24h", "cardholder_age"
]

# ======= Title of the page =======
st.set_page_config(layout="wide")
st.title("💳 Credit Card Fraud Transaction")

basic_EDA, feature_engineering_and_predict = st.tabs(
    ["EDA", "Feature Engineering & Predict"]
)

# Load CSS
with open("style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

with basic_EDA:
    overall_metric(df)
    st.write("-------------")    
    
    c1, _, c2 = st.columns([3, 0.02, 1])

    with c1:
        st.write("### Overview")
        overview_db(df)

    with c2:
        st.write("Target to predict: `is_fraud`")
        target_distribution(df, targ_col = 'is_fraud')

    st.write("-------------")
    st.write("### Advanced Analytic")
    get_all_distribution(df, num_cols)

    a1, _, a2 = st.columns([2, 0.01, 3])
    with a1:
        multivariate_report(df, num_cols)
    with a2:
        binary_feature_report(df)
    
    merchant_report(df)

with feature_engineering_and_predict:
    
    enhancing_db = feature_eningineering_enhancing(df)
    review_new_results(enhancing_db)
    st.write("-----------")
    
    with st.expander("Model comparison", expanded = True):
        st.write("Both scenarios will use the same ML-model which selected in the left-pannel")
        c1, c2 = st.columns(2)
        with c1:
            with st.expander("#### Without using resampling", expanded = True):
                enhancing_db['is_fraud'] = enhancing_db['is_fraud'].astype(int)
                X_labeled, y = label_encoding(enhancing_db)
                X_train, X_test, y_train, y_test, feature_names, scaler = split_and_scaling(X_labeled, y)
                
                # Split column
                a, b, c = st.columns(3)
                with a:
                    model_name = st.selectbox("Model name", 
                                            ["Logistic Regression", "Random Forest", "Decision Tree", "KNN", "XGBoost"])

                # Model selected
                if model_name == "Logistic Regression":
                    with b:
                        param1 = st.number_input("C (inverse reg.)", 0.0001, 1000.0, 1.0, step=0.1)
                    with c:
                        param2 = st.slider("L1 ratio (0=L2, 1=L1)", 0.0, 1.0, 0.0, step=0.1)

                elif model_name == "Random Forest":
                    with b:
                        param1 = st.slider("n_estimators", 50, 500, 100, step=50)
                    with c:
                        param2 = st.slider("max_depth", 1, 30, 10)

                elif model_name == "Decision Tree":
                    with b:
                        param1 = st.slider("max_depth", 1, 30, 5)
                    with c:
                        param2 = st.selectbox("criterion", ["gini", "entropy"])

                elif model_name == "KNN":
                    with b:
                        param1 = st.slider("n_neighbors (K)", 1, 50, 5)
                    with c:
                        param2 = st.selectbox("weights", ["uniform", "distance"])
                
                else:
                    with b:
                        param1 = st.slider("n_estimators", 50, 500, 100, step=50)
                    with c:
                        param2 = st.slider("learning_rate", 0.01, 0.5, 0.1)

                # Fit & train
                clf = get_clf_model(model_name, param1, param2)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                # Charts
                cm, fi = st.columns([2, 3])
                with cm:
                    get_confusion_matrix(y_test, y_pred)

                with fi:
                    importance_df = get_importance_features_table(model_name, clf, X_test, y_test, feature_names)
                    get_feature_importance_chart(importance_df)

        with c2:
            with st.expander("#### Using resampling", expanded = True):
                l, r = st.columns(2)
                with l:
                    resampling_method = st.selectbox("Resampling method", ["Undersampling", "Oversampling"])
                with r:
                    if resampling_method == "Undersampling":
                        sampling_choices = ["TomeLinks", "RandomUnderSampler"]
                    else:
                        sampling_choices = ["SMOTE", "RandomOverSampler"]
                    methodology_name = st.selectbox("Resampling method name", sampling_choices)

                sampler = get_sampler(resampling_method, methodology_name)
                X_resampled, y_resampled = sampler.fit_resample(X_labeled, y)
                X_train_new, X_test_new, y_train_new, y_test_new, feature_names, scaler = split_and_scaling(X_resampled, y_resampled)

                clf_resample = get_clf_model(model_name, param1, param2)
                clf_resample.fit(X_train_new, y_train_new)

                y_pred_new = clf_resample.predict(X_test_new)
                cm, fi = st.columns([2, 3])

                with cm:
                    get_confusion_matrix(y_test_new, y_pred_new)

                with fi:
                    importance_df = get_importance_features_table(
                        model_name, clf_resample, X_test_new, y_test_new, feature_names
                    )
                    get_feature_importance_chart(importance_df, key = "resampling")
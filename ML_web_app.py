# Libraries and packages used
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

def main():
    #+========================================================+
    #|                     FUNCTIONS                          |
    #+========================================================+
    @st.cache(persist = True)
    def load_data():
        data = pd.read_csv('mushrooms.csv')

        # Encode target labels with value between 0 and n_classes-1
        label =  LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache(persist=True)
    def split(df):
        y = df.type
        x = df.drop(columns=['type'])
        return train_test_split(x,y, test_size=0.3,random_state=0)

    def show_results(metrics_list):
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)

        precision = precision_score(y_test, y_pred, labels = class_names)
        recall = recall_score(y_test, y_pred, labels = class_names)

        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision.round(2))
        st.write("Recall: ", recall.round(2))

        # Plot metrics depending on "metrics_list" selected
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curves")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

    #+========================================================+
    #|                   INITIALIZATION                       |
    #+========================================================+
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonous']

    #+========================================================+
    #|                   VISUAL COMPONENTS                    |
    #+========================================================+
    st.title("🖥️ Binary classification web App")
    st.sidebar.title("⚙️ Classification options")
    st.write("""Machine learning application to **test different types of classifiers
                and the combination of their hyperparameters** with the Sklearn library.
                The complete code can be found in this [GitHub repository](https://github.com/Alejandro-ZZ/ML-web-app)""")
    st.write("""A binary classification is performed with the [Kaggle "mushrooms" dataset]
                (https://www.kaggle.com/uciml/mushroom-classification). The objective
                is to **classify mushrooms as either edible or poisonous mushrooms**.""")

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)

    st.sidebar.subheader("Choose classifier and metrics")
    classifier = st.sidebar.selectbox("📁 Classifier",("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))
    metrics = st.sidebar.multiselect("📊 What metric to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    st.sidebar.subheader("Model Hyperparameters")

    #+========================================================+
    #|                      CLASSIFIERS                       |
    #+========================================================+
    if classifier == 'Support Vector Machine (SVM)':
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step = 0.01, key = 'C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key = 'kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficiente", ("scale", "auto"), key = 'gamma')

        if st.sidebar.button("▶️ Classify", key = 'classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C = C, kernel = kernel, gamma = gamma)
            show_results(metrics)

    if classifier == 'Logistic Regression':
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step = 0.01, key = 'C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key = 'max_iter')

        if st.sidebar.button("▶️ Classify", key = 'classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C = C, max_iter = max_iter)
            show_results(metrics)

    if classifier == 'Random Forest':
        n_estimators = st.sidebar.number_input("Number of trees in the forest", 100, 500, step = 10, key = 'n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step = 1, key = 'max_depth')
        bootstrap = st.sidebar.radio("Bootstrap sample when biulding trees", ('True', 'False'), key = 'bootstrap')

        if st.sidebar.button("▶️ Classify", key = 'classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators = n_estimators , max_depth = max_depth, bootstrap = bootstrap, n_jobs = -1)
            show_results(metrics)

if __name__ == '__main__':
    main()

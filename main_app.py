import lime.lime_tabular
from catboost import CatBoostClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_curve,auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import shap
from streamlit.components.v1 import html
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import plotly.express as px
import plotly.graph_objs as go
from pdpbox import pdp, info_plots
from sklearn.inspection import partial_dependence


st.set_page_config(layout="wide")
st.title("Classification Problem Dashboard")
st.header(":orange[Obesity Dataset] Dashboard",divider='orange')
st.subheader("Interactive tool for :green[visualizing and predicting obesity probability] based on lifestyle and health data",divider='green')
df = pd.read_csv("obesity_classification.csv")
cols = ['Height', 'Weight', 'family_history_with_overweight', 'SCC',
       'MTRANS_Walking', 'FAVC_z', 'FCVC_minmax', 'NCP_z', 'CAEC_minmax',
       'CH2O_minmax', 'FAF_minmax', 'TUE_z', 'CALC_z', 'Age_bin_minmax','NObeyesdad']

target = 'NObeyesdad'

rename = {}
for i in range(1,len(cols)):
    rename[cols[i-1]] = f"C_{i}"

rename["NObeyesdad"] = "Target"

df.rename(rename,axis=1,inplace=True)
X = df.drop("Target",axis=1)
y = df["Target"]
st.sidebar.header('Options')
options = ['Data Overview', 'Model Building and Evaluation', 'Make Predictions']
selection = st.sidebar.selectbox("Go to", options)

meaning = {}
m = ["Height","Weight","Has a family member suffered or suffers from overweight?",
         "Do you eat high caloric food frequently?","Do you usually eat vegetables in your meals?",
         "How many main meals do you have daily?","Do you eat any food between meals?","Do you smoke?",
         "How much water do you drink daily?","Do you monitor the calories you eat daily?","How often do you have physical activity?",
         "How much time do you use technological devices such as cell phone, videogames, television, computer and others?",
         "How often do you drink alcohol?","Which transportation do you usually use?","Obesity level - Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, and Obesity Type III"]
    
for i in range(len(df.columns)):
    meaning[df.columns[i]]=m[i]

class_labels = ["Insufficient Weight", "Normal Weight", "Overweight Level I", "Overweight Level II", "Obesity Type I", "Obesity Type II", "Obesity Type III"]
if selection == 'Data Overview':
    st.subheader(':orange[Data] Overview')
    
    col1,col2=st.columns(2)
    with col1:
        st.sidebar.write("### Data Overview selected.")
        st.dataframe(df,height=670)
        st.link_button("Dataset Reference","https://www.kaggle.com/datasets/ikjotsingh221/obesity-risk-prediction-cleaned/data")
    with col2:
        st.write("### Column descriptions")
        st.table(meaning)
    st.divider()
    st.write(":blue[Quick glance at the dataset attributes] and their role in determining obesity levels.")
    st.write("Shape of the dataset : ",df.shape)
    
    s = df.describe(include='all')
    s.loc["Type"] = [df[col].dtype for col in df]
    s.loc["Missing Values"] = [df[col].isnull().sum() for col in df]
    st.dataframe(s)
    st.write(df.isnull().sum().sum(),"missing values in the dataset.")
    col1,col2 = st.columns((8,8))
    with col1:
        col = st.selectbox('Select Column to View Distribution', df.columns)
        fig = px.histogram(df,x=col,marginal='rug',nbins=50,title=f"Distribution of {col}")
        st.plotly_chart(fig)

    with col2:
        selected_features = st.multiselect('Select Features for Pair Plot', df.columns)
        if selected_features:
            st.write('Pair Plot:')
            pair_plot_data = df[selected_features]
            fig = px.scatter_matrix(pair_plot_data,dimensions=list(pair_plot_data.columns))
            st.plotly_chart(fig)
    st.write('### Correlation Heatmap:')
    corr = X.corr()
    fig = px.imshow(corr,text_auto = True, aspect='auto',title='Correlation Heatmap')
    st.plotly_chart(fig)

    numerical_cols = df.select_dtypes(include=['float','int']).columns
    if len(numerical_cols) > 0:
        st.write('### Violin Plots for Numerical Features:')
        for col in numerical_cols[:-1]:
            fig = px.violin(df,y=col,x='Target',box=True,title=f"Distribution of {col} by Target")
            st.plotly_chart(fig)

elif selection == 'Model Building and Evaluation':
    
    st.sidebar.write("Model Building and Evaluation selected.")
    st.subheader(':green[Model] Building and Evaluation')

    model_type = st.selectbox('Select Model Type', ['RandomForestClassifier' ,'XGBClassifier', 'LGBMClassifier','CatBoostClassifier'])
    st.write("Select test size.")
    test_size = st.slider('Test Size', min_value=0.05, max_value=0.50, value=0.25, step=0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    st.session_state['X_train'] = X_train
    st.session_state["X_test"] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test
    st.session_state["y"] = y
    st.session_state["X"] = X
    # Initialize the model
    if model_type == 'RandomForestClassifier':
        n_estimators = st.slider('Number of Estimators', 10, 100, 50)
        max_depth = st.slider('Max Depth', 1, 20, 5)
        criterion = st.selectbox('Criterion',['gini','entropy','log_loss'])
        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion ,max_depth=max_depth,random_state=42)
    
    elif model_type == 'XGBClassifier':
        max_depth = st.slider('Max Depth', 1, 20, 5)
        n_estimators = st.slider('Number of Estimators', 10, 500, 100, step=10)
        learning_rate = st.slider('Learning Rate', 0.01, 1.0, 0.1, step=0.01)
        model = XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)

    elif model_type == 'GradientBoostingClassifier':
        n_estimators = st.slider('Number of Estimators', 10, 500, 100, step=10)
        max_depth = st.slider('Max Depth', 1, 20, 5)
        learning_rate = st.slider('Learning Rate', 0.01, 1.0, 0.1, step=0.01)
        model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)

    elif model_type == 'LGBMClassifier':
        n_estimators = st.slider('Number of Estimators', 10, 500, 100, step=10)
        max_depth = st.slider('Max Depth', 1, 20, 5)
        learning_rate = st.slider('Learning Rate', 0.01, 1.0, 0.1, step=0.01)
        model = LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)

    elif model_type == 'CatBoostClassifier':
        n_estimators = st.slider('Number of Estimators', 10, 500, 100, step=10)
        max_depth = st.slider('Max Depth', 1, 20, 5)
        learning_rate = st.slider('Learning Rate', 0.01, 1.0, 0.1, step=0.01)
        model = CatBoostClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)


    train_model = st.button('Train Model')
    st.write(":red[*LGBMClassifier and GradientBoostingClassifer might take more than 40 seconds to train]")
    if train_model:
        st.write(':blue[Training the model], please wait...')
        if model_type != 'RandomForestClassifier' and model_type!="LGBMClassifier":
               model.fit(X_train, y_train,early_stopping_rounds=10,
        eval_set=[(X_test, y_test)])
        else:
           model.fit(X_train, y_train)
        st.session_state['model'] = model
        predictions = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:,1]
        st.subheader(f":green[Model Performance Metrics on {model_type}]")
        accuracy = accuracy_score(y_test, predictions)
        st.metric(label="Accuracy", value=f"{accuracy*100:.2f}%")
        c1,c2 = st.columns(2)
        with c1:
            st.write('Classification Report:')
            report = pd.DataFrame(classification_report(y_test, predictions, output_dict=True)).transpose()
            st.dataframe(report,width=500)
        with c2:
            st.write('Confusion Matrix:')
            cm = confusion_matrix(y_test, predictions)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('True Labels')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
        
        try:
            st.markdown("## :orange[Feature Importance and Interpretability]")
            st.write("### Feature Importance Chart")
            st.write("Bar chart highlighting the relative importance of each feature in the model.")

            # Assuming your model has a feature_importances_ attribute
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            names = [X_train.columns[i] for i in indices]

            # Creating the bar plot
            fig, ax = plt.subplots()
            sns.barplot(x=importances[indices], y=names, palette="viridis", ax=ax)
            ax.set_title("Feature Importance")
            ax.set_xlabel("Relative Importance")
            st.pyplot(fig)
        except:
            pass

        st.markdown("## Model Insights")
        st.write("### Visual aids for understanding individual prediction rationales")
        c1,c2 = st.columns(2)
        with c1:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values, 
                feature_names=X_train.columns.tolist(), 
                class_names=class_labels, 
                mode='classification'
            )
            exp = explainer.explain_instance(X_test.values[0], model.predict_proba, num_features=5)
            st.write('LIME Explanation')
            exp_html = exp.as_html()
            html(exp_html,height=500)
        with c2:
            shap.initjs()  # Initialize JavaScript for SHAP in the notebook
            shap_explainer = shap.TreeExplainer(model)
            st.write("SHAP explanation")
            shap_values_full = shap_explainer.shap_values(X_test)
            
            plt.figure()
            shap.summary_plot(shap_values_full, X_test,plot_type='dot')

            # Save the current Matplotlib figure in a variable and pass it to st.pyplot
            fig = plt.gcf()
            st.pyplot(fig)
            

    
elif selection == 'Make Predictions':
    st.sidebar.write("Fill in the inputs to make predictions.")
    st.subheader(':violet[Making Predictions] using trained Model')
    if "model" in st.session_state:
        model = st.session_state["model"]
        Y = st.session_state["y"]
        X_train = st.session_state["X_train"]
        c1,c2 = st.columns(2)
        with c1:
            inputs = []
            
            for col in df.columns[:-1]:
                val = st.slider(meaning[col],min_value=df[col].min(),max_value=df[col].max())
                inputs.append(val)
        with c2:
            prediction=model.predict([inputs])
            col1,col2 = st.columns(2)
            with col1:
                st.markdown("### Model Prediction : <strong style='color:tomato;'>{}</strong>".format(Y[prediction[0]]),unsafe_allow_html=True)
            probs = model.predict_proba([inputs])

            probability = probs[0][prediction[0]]
            
            with col2:
                try:
                    st.metric(label="Model Confidence",value="{:.2f} %".format(probability*100),delta="{:.2f} %".format((probability-0.5)*100))

                    explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train),mode='classification',
                                                        class_names=class_labels,
                                                        feature_names=list(X_train.columns))
                    explanation = explainer.explain_instance(np.array(inputs),model.predict_proba,
                                                    num_features = len(list(X_train.columns[:-1])),
                                                    top_labels = 3)

                    interpretation_fig = explanation.as_pyplot_figure(label=prediction[0])
                    st.pyplot(interpretation_fig,use_container_width=True)
                except:
                    probaby = probability[0]
                    formatted_probability = "{:.2f} %".format(probaby * 100)
                    formatted_delta = "{:.2f} %".format((probaby - 0.5) * 100)
                    st.metric(label="Model Confidence",value=formatted_probability,delta=formatted_delta)

                    explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train),mode='classification',
                                                        class_names=class_labels,
                                                        feature_names=list(X_train.columns))
                    explanation = explainer.explain_instance(np.array(inputs),model.predict_proba,
                                                    num_features = len(list(X_train.columns[:-1])),
                                                    top_labels = 3)
                    print(prediction[0])
                    interpretation_fig = explanation.as_pyplot_figure(label=prediction[0][0])
                    st.pyplot(interpretation_fig,use_container_width=True)


    else:
        st.write("Run the Model Building and Evaluation step.")
    

# Function to display contact details with a custom message
def display_contact_details_with_love(details):
    st.markdown("---")  # Horizontal line for separation
    st.markdown(f"### Made with :heart: by {details['Full Name']}")
    details.pop('Full Name')
    for detail, value in details.items():
        st.markdown(f"[{detail}]({value})")

# Your personal details
personal_details = {
    'Full Name': 'Chekuri Viroopaksh',
    'LinkedIn': 'https://www.linkedin.com/viroopaksh-chekuri',
    'GitHub': 'https://github.com/VIROOPAKSHC'
}
with st.sidebar:
    display_contact_details_with_love(personal_details)

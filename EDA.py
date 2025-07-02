import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_wine, load_breast_cancer
st.title('Data Analysis Application')
st.subheader('This ia simple data analysis application created for data exploration and visualization')

df= sns.load_dataset('tips')
# List of available datasets
dataset_options = ['iris', 'titanic', 'tips', 'diamonds', 'Wine (sklearn)', 'Breast Cancer (sklearn)', 'Upload your own']
dataset_options = ['iris', 'titanic', 'tips','diamonds','load_wine','load_breast_cancer','Upload your own']

# Dropdown to select dataset
selected = st.selectbox("Select a dataset", dataset_options)

df = None

if selected == 'Upload your own':
    uploaded_file = st.file_uploader("Upload your file", type=["csv","xlsx"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
else:
    if selected == 'load_wine':
        data = load_wine(as_frame=True)
        df = data.frame
        st.success("Loaded Wine dataset from sklearn.")
    elif selected == 'load_breast_cancer':
        data = load_breast_cancer(as_frame=True)
        df = data.frame
        st.success("Loaded Breast Cancer dataset from sklearn.")
    else:
        df = sns.load_dataset(selected)
        st.success(f"Loaded {selected} dataset from seaborn.")

if df is not None:
    st.write("Preview of the dataset:")
    st.dataframe(df.head( ))
    # display number of rows and columns
    st.subheader('Dataset Information')
    st.write('Number of rows',df.shape[0])
    st.write('Number of columns:',df.shape[1])
    # display the column names of selected data with their data types
    st.write('Column Names and Data types:')
    st.dataframe(df.dtypes.rename("Data Type").reset_index().rename(columns={"index": "Column Name"}).set_index("Column Name"))
    #print null values if those are greater than zeros
    null_counts = df.isnull().sum()
    null_columns = null_counts[null_counts > 0]
    if not null_columns.empty:
        st.warning("Columns with null values:")
        st.table(null_columns.rename("Null Count"))
    # display summary statistics of the selected data
    st.write('Summary Statistics',df.describe())
    #select the specif columns for X or y axis from the dataset and also select the plot type
    x_axis= st.selectbox('Select X-axis',df.columns)
    y_axis = st.selectbox('Select Y-axis',df.columns)
    plot_type = st.selectbox('Select Plot type',['line','scatter','hist','bar','box'])
    # plot the data
    if plot_type == 'line':
        st.line_chart(df[[x_axis,y_axis]])
    elif plot_type == 'scatter':
        st.scatter_chart(df[[x_axis,y_axis]])
    elif plot_type == 'hist':
         df[x_axis].plot(kind = 'hist')
         st.pyplot()
    elif plot_type == 'bar':
         st.bar_chart(df[[x_axis,y_axis]])
    elif plot_type == 'box':
        df[[x_axis,y_axis]].plot(kind='box')
        st.pyplot()
    elif plot_type =='kde':
        df[[x_axis,y_axis]].plot(kind ='kde')
        st.pyplot()
if df is not None:
    #create a heatmap
    st.subheader('Heatmap')

    #select the columns which are numeric
    numeric_columns = df.select_dtypes(include = np.number).columns
    corr_matrix = df[numeric_columns].corr()
    from plotly import graph_objects as go
    st.write("Correlation Matrix:")
    st.dataframe(corr_matrix)

    heatmap_fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,  # fixed typo here
            colorscale='Viridis'
        )
    )
    st.plotly_chart(heatmap_fig)

    #Create a pairplot
    st.subheader('Pairplot')
    hue_columns = st.selectbox('Select a column to be used as hue',df.columns)
    st.write("Generating pairplot. This may take a moment for large datasets.")
    st.pyplot(sns.pairplot(df,hue = hue_columns))



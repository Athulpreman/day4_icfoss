
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib
import seaborn as sns

#Remove Warnings
st.balloons()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Dataset")

df = pd.read_csv("titanic.csv")

titanic = df.head(20)

st.table(titanic)
st.header("Visualisation Using Seaborn")

#bar plot
st.subheader("Bar Plot")
titanic.plot(kind='bar')
st.pyplot()

#Displot
st.subheader("Displot")
sns.displot(titanic['Survived'])
st.pyplot()

#joinplot
st.subheader("JointPlot")
sns.jointplot(x='Age',y='Survived',data=titanic,kind='scatter')
st.pyplot()

#pairplot
st.subheader("Pairplot")
sns.pairplot(titanic,hue='Sex',palette='rainbow')
st.pyplot()

#Rugplot
st.subheader("Rugplot")
sns.rugplot(titanic['Survived'])
st.pyplot()

#Correation
st.subheader("Heatmap")
sns.heatmap(titanic.corr(),cmap='coolwarm',annot=True)
st.pyplot()

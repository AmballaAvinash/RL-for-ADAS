import plotly.express as px
import pandas as pd



# Compare all algorithms for a single exmaple. Repeat for all the expamples and torcs
df1 = pd.read_csv("output_plots/Custom_CNN/CSV/A2C_Single_V_ex1.csv")
df2 = pd.read_csv("output_plots/Custom_CNN/CSV/A2C_Single_V_ex2.csv")

Episode = df1["Episode"].iloc[:].values
MeanReward1 = df1["Mean Reward"].iloc[:].values
MeanReward2 = df2["Mean Reward"].iloc[:].values

df = pd.DataFrame({"Episode" : Episode , "Mean Reward ex1": MeanReward1 , "Mean Reward ex2": MeanReward2})
fig = px.line(df, x="Episode", y=df.columns[1:], labels = {}, title="Reward Curve") 
fig.show()
fig.write_image("test.eps")
    
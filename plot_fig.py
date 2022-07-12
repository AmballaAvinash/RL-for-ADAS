import plotly.express as px
import pandas as pd

# Compare all algorithms for a single exmaple. Repeat for all the expamples and torcs

df1 = pd.read_csv("output_plots/Custom_CNN/CSV/PG_ex1.csv")
df2 = pd.read_csv("output_plots/Custom_CNN/CSV/PG_single_network_ex1.csv")
df3 = pd.read_csv("output_plots/Custom_CNN/CSV/A2C_Single_V_ex1.csv")
df4 = pd.read_csv("output_plots/Custom_CNN/CSV/A2C_single_network_ex1.csv")
df5 = pd.read_csv("output_plots/Custom_CNN/CSV/Duelling_DQN_single_Q_ex1.csv")
df6 = pd.read_csv("output_plots/Custom_CNN/CSV/random_agent_ex1.csv")
df7 = pd.read_csv("output_plots/Custom_CNN/CSV/intelligent_agent_ex1.csv")


Episode = df1["Episode"].iloc[:].values
MeanReward1 = df1["Mean Reward"].iloc[:].values
MeanReward2 = df2["Mean Reward"].iloc[:].values
MeanReward3 = df3["Mean Reward"].iloc[:].values
MeanReward4 = df4["Mean Reward"].iloc[:].values
MeanReward5 = df5["Mean Reward"].iloc[:].values
MeanReward6 = df6["Mean Reward"].iloc[:].values
MeanReward7 = df7["Mean Reward"].iloc[:].values


std1 = df1["Std"].iloc[:].values
std2 = df2["Std"].iloc[:].values
std3 = df3["Std"].iloc[:].values
std4 = df4["Std"].iloc[:].values
std5 = df5["Std"].iloc[:].values
std6 = df6["Std"].iloc[:].values
std7 = df7["Std"].iloc[:].values


df = pd.DataFrame({"Episode" : Episode ,"PG Multi-agent": MeanReward1,
                    "PG Single-agent": MeanReward2, "A2C Mutli-agent" :  MeanReward3 , 
                    "A2C Single-agent": MeanReward4, "Duelling DQN Single-agent": MeanReward5,                   
                    "Random-agent": MeanReward6, "Intelligent-agent" : MeanReward7,
                   })

# improve line width, color, legend size , lable names ND sizee
fig = px.line(df, x="Episode", y=df.columns[1:],title="Reward Curve") 


# lines width (curren colors are ok)
fig.update_traces(line=dict(width=3))


# legend size 
fig.update_layout(
    title=dict(
        text='<b>Reward Curve</b>',
        x=0.4,
        y=0.95,
        font=dict(
            family="Arial",
            size=42,
            color='#000000'
        )
    ),
    xaxis_title="<b>Episode</b>",
    yaxis_title='<b>Mean Reward</b>',
    font=dict(
        family="Courier New, Monospace",
        size=42,
        color='#000000'
    ),
    legend=dict(itemsizing='constant')
)


fig.show()
fig.write_image("ex1_out.pdf",width=2160, height=1080)
    
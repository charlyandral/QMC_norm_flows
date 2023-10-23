#%%
import pandas as pd
import numpy as np
import seaborn as sns
#%%

df = pd.read_pickle("archi_test.zip")
df["method"].replace("QMCSobol_True","QMC",inplace=True)
df["variable"].replace("mean_0","x1",inplace=True)
df = df.rename(columns={'variable': 'test function'})
df["error"] = df["value"].apply(np.abs)
df_plot = df[df["test function"] != "mean_1"]
# %%
df_plot = df[df["test function"] != "mean_1"]
g = sns.catplot(data= df_plot, y = "error",x= "archi",hue = "method",col = "test function",kind = "box",height=3, aspect=1.3)
g.set_xticklabels(rotation = 20)
g.set(yscale="log")
g.savefig("./Latex/analyze_archi.pdf")

df_plot.groupby(["id"])["loss"].mean()
# %%
df_grp = df.groupby(['method',"test function","flow"])["error"].mean()
out_df = df_grp.loc["MC"]/ df_grp.loc["QMC"]
print(out_df.to_latex())
# %%

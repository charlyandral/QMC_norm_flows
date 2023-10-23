#%%
import numpy as np
import seaborn as sns
import pandas as pd
# %%
df = pd.read_pickle("all.zip")

df_mean0 = df[df["variable"] == "sin(x1)*x1^4"]
df_mean0.reset_index()
df_mean0["dim"] = (df_mean0.reset_index().index // (df_mean0.shape[0]//9)) + 2

out_df = pd.DataFrame()
out_df["sin"] = df_mean0[df_mean0["method"] == "MC"].groupby("dim")["value"].std()/df_mean0[df_mean0["method"] != "MC"].groupby("dim")["value"].std()


df_mean0 = df[df["variable"] == "mean_0"]
df_mean0.reset_index()
df_mean0["dim"] = (df_mean0.reset_index().index // (df_mean0.shape[0]//9)) + 2

out_df["mean"] = df_mean0[df_mean0["method"] == "MC"].groupby("dim")["value"].std()/df_mean0[df_mean0["method"] != "MC"].groupby("dim")["value"].std()
print(out_df.T.to_latex(float_format="%.2f").replace('\\toprule', '\\hline').replace('\\midrule', '\\hline').replace('\\bottomrule', ''))


df_mean0 = df[df["variable"] == "sin(x1)*x1^4"]
df_mean0.reset_index()
df_mean0["dim"] = (df_mean0.reset_index().index // (df_mean0.shape[0]//9)) + 2

out_df = pd.DataFrame()
out_df["sin"] = df_mean0[df_mean0["method"] == "MC"].groupby("dim")["value"].apply(lambda x: np.mean(np.abs(x)))/df_mean0[df_mean0["method"] != "MC"].groupby("dim")["value"].apply(lambda x: np.mean(np.abs(x)))


df_mean0 = df[df["variable"] == "mean_0"]
df_mean0.reset_index()
df_mean0["dim"] = (df_mean0.reset_index().index // (df_mean0.shape[0]//9)) + 2

out_df["mean"] = df_mean0[df_mean0["method"] == "MC"].groupby("dim")["value"].apply(lambda x: np.mean(np.abs(x)))/df_mean0[df_mean0["method"] != "MC"].groupby("dim")["value"].apply(lambda x: np.mean(np.abs(x)))

print(out_df.T.to_latex(float_format="%.2f").replace('\\toprule', '\\hline').replace('\\midrule', '\\hline').replace('\\bottomrule', ''))

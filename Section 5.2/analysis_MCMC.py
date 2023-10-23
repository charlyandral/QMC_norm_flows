#%%
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy.random as npr
from sklearn.linear_model import LinearRegression
import numba as nb
import arviz as az
jax.config.update("jax_enable_x64", True)
df_all = pd.read_pickle("archi_test.zip")
from tqdm import tqdm
import seaborn as sns
#%%

npr.seed(0)
@nb.njit()
def indep_MH_nf(proposals, weights):
    n_iter = proposals.shape[0]
    out = np.zeros_like(proposals)
    out[0] = proposals[0]
    accepted = 0
    copies = np.zeros((n_iter))
    pos_copie = 0
    current_weight = weights[0]
    for i in range(1, n_iter):
        alpha = min(weights[i] / current_weight, 1)
        if npr.random() < alpha:
            out[i] = proposals[i]
            accepted += 1
            pos_copie += 1
            current_weight = weights[i]
            copies[pos_copie] += 1
        else:
            out[i] = out[i - 1]
            copies[pos_copie] += 1
    return out, copies



@nb.njit()
def compute_chain(
    points: np.array, weights: np.array, alpha: float = 1
):
    n_iter = points.shape[0]
    copies = np.ones(n_iter, dtype=np.int_)
    ratios = weights
    kappa = alpha * n_iter / ratios.sum()
    for i in range(n_iter):
        copies[i] = int(kappa * ratios[i]) + (npr.rand() < (kappa * ratios[i] % 1))
    n_tot = np.cumsum(copies)
    n_tot = np.concatenate((np.zeros(1, dtype=np.int_), n_tot))
    out = np.zeros((n_tot[-1], points.shape[1]))
    for i in range(0, n_iter):
        out[n_tot[i] : n_tot[i + 1]] = points[i]
    return out, copies



def analyze(df_all):
    out_df = []
    for index,row in tqdm(df_all.iterrows()):
        sample = np.array(row.samples)
        weights = np.array(row.weights)
        chain_imc,_ = compute_chain(sample,weights)
        chain_imrth,_ = indep_MH_nf(sample,weights)
        ess_imc =az.ess(az.convert_to_dataset(chain_imc[np.newaxis, :])).x.to_numpy()
        ess_imrth = az.ess(az.convert_to_dataset(chain_imrth[np.newaxis, :])).x.to_numpy()
        out_df.append({"ESS":ess_imc[0],"mean": np.mean(chain_imc[:,0]), "method": row.method, "algo" : "iIMC"} )
        out_df.append({"ESS":ess_imrth[0], "mean" : np.mean(chain_imrth[:,0]),"method": row.method, "algo" : "iIMRTH"} )  
        out_df.append({"ESS": 0, "mean" : np.average(sample[:,0], weights=weights),"method": row.method, "algo" : "IS"})
        
    df = pd.DataFrame(out_df)
    return df
df =  analyze(df_all)
df["error"] = np.abs(df["mean"])
#%%
fig,ax = plt.subplots(1,2,figsize=(8,3.5))
sns.boxplot(data=df[df["algo"] != 'IS'], hue="method", y="ESS",x = "algo",ax = ax[0])
handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend().remove()
sns.boxplot(data=df, hue="method", y="error",x = "algo",ax=ax[1])
ax[1].set_yscale("log")
ax[1].legend().remove()
ax[1].set_ylabel("Error", labelpad=-2)
fig.legend(handles, labels,fontsize=7)
fig.savefig("mcmc.pdf")
# %%

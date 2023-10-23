
#%%
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
jax.config.update("jax_enable_x64", True)
df_all = pd.read_pickle("archi_test.zip")
#%%

fun = lambda x: x[:, 0]
fun2 = lambda x: jnp.sin(x[:, 0] * 10) * x[:, 0] ** 4
begin = 6
end = 17
n_points_inter = 20
x_arr = jnp.logspace(begin, end, (end - begin) * n_points_inter + 1, base=2, dtype=int)

def cumulative_plot(samples, weights, x_arr, function):
    """
    Plot the cumulative mean of a function of the samples
    """
    out = []
    for j in x_arr:
        mean = jnp.average(a=function(samples[:j]), weights=weights[:j])
        out.append(mean)
    return out


cumulative_plot_vmap = jax.vmap(cumulative_plot, in_axes=(0, 0, None, None))



title = ["x1","sin(x1*10)*x1^4"]

def analyze(list_df, x_arr, funs):
    fig, ax = plt.subplots(1,2,sharey=0, figsize = (6,3.5))
    for i,fun in enumerate(funs):
        for name, df in list_df:
            arr_sample = jnp.asarray(np.stack(df.samples.to_numpy()))
            arr_weight = jnp.asarray(np.stack(df.weights.to_numpy()))
            out_vmap = cumulative_plot_vmap(arr_sample, arr_weight, x_arr, fun)
            mean = [jnp.mean(jnp.abs(out_vmap[i][:])) for i in range(len(out_vmap))]
            ax[i].plot(x_arr, mean, label=name,linewidth = 0.8)
            ax[i].set_title(title[i])
            mean = jnp.asarray(mean)

        m = -0.5
        y1 = np.exp2(-1) * x_arr**m
        y2 = np.exp2(1) * x_arr**m
        y4 = np.exp2(-3) * x_arr**m
        

        # Plot the lines
        ax[i].plot(x_arr, y1, linestyle="--", linewidth=.7, alpha=0.5, c="grey")
        ax[i].plot(x_arr, y2, linestyle="--", linewidth=.7, alpha=0.5, c="grey")
        ax[i].plot(x_arr, y4, linestyle="--", linewidth=.7, alpha=0.5, c="grey")

        ax[i].loglog(base=2)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels,fontsize=7)
    fig.supxlabel("Iterations")
    fig.supylabel("Error")
    fig.savefig("loglogplot.pdf")
funs = [fun,fun2]

analyze(df_all.groupby("method"), x_arr, funs)

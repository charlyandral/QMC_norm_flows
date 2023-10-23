#%%
from flowMC.sampler.MALA import MALA
import jax
import jax.numpy as jnp  # JAX NumPy
from jax.scipy.special import logsumexp
import numpy as np

from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.nfmodel.utils import *
from flowMC.sampler.MALA import MALA
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys
import scipy.stats as stats
import pandas as pd
from joblib import Parallel, delayed
import seaborn as sns
import scipy.stats.qmc as qmc
from tqdm import tqdm
jax.config.update("jax_enable_x64", True)
import corner
import matplotlib.pyplot as plt



class QMC_flow:
    def __init__(
        self, n_dim, target, QMC_engine, seed_training_flow, model, n_loop_training=40
    ):
        self.n_dim = n_dim
        self.target = target
        self.QMC_engine = QMC_engine
        self.seed_training_flow = seed_training_flow
        self.target_vmap = jax.jit(jax.vmap(lambda x: self.target(x, jnp.zeros(n_dim))))
        self.model = model
        self.n_loop_training = n_loop_training

    def trainning_flow(self):
        n_dim = self.n_dim
        n_chains = 100
        n_loop_training = self.n_loop_training
        n_loop_production = 5
        n_local_steps = 100
        n_global_steps = 100
        learning_rate = 0.001
        momentum = 0.9
        num_epochs = 30
        batch_size = 10000

        data = jnp.zeros(n_dim)

        rng_key_set = initialize_rng_keys(n_chains, 41)

        initial_position = (
            jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1
        )

        MALA_Sampler = MALA(self.target, True, {"step_size": 0.1})

        print("Initializing sampler class")

        nf_sampler = Sampler(
            n_dim,
            rng_key_set,
            jnp.zeros(n_dim),
            MALA_Sampler,
            self.model,
            n_loop_training=n_loop_training,
            n_loop_production=n_loop_production,
            n_local_steps=n_local_steps,
            n_global_steps=n_global_steps,
            n_chains=n_chains,
            n_epochs=num_epochs,
            learning_rate=learning_rate,
            momentum=momentum,
            batch_size=batch_size,
            use_global=True,
        )

        nf_sampler.sample(initial_position, data)
        self.nf_sampler = nf_sampler

        summary = nf_sampler.get_sampler_state(training=True)
        chains, log_prob, local_accs, global_accs, loss_vals = summary.values()
        print(
            "chains shape: ",
            chains.shape,
            "local_accs shape: ",
            local_accs.shape,
            "global_accs shape: ",
            global_accs.shape,
        )

        chains = np.array(chains)
        loss_vals = np.array(loss_vals)

        # Plot one chain to show the jump
        plt.figure(figsize=(6, 6))
        axs = [plt.subplot(2, 2, i + 1) for i in range(4)]
        plt.sca(axs[0])
        plt.title("2 chains")
        plt.plot(chains[0, :, 0], chains[0, :, 1], alpha=0.5)
        plt.plot(chains[1, :, 0], chains[1, :, 1], alpha=0.5)
        #plt.xlabel("$x_1$")
        #plt.ylabel("$x_2$")

        plt.sca(axs[1])
        plt.title("NF loss")
        plt.plot(loss_vals.reshape(-1))
        plt.xlabel("iteration")

        plt.sca(axs[2])
        plt.title("Local Acceptance")
        plt.plot(local_accs.mean(0))
        plt.xlabel("iteration")

        plt.sca(axs[3])
        plt.title("Global Acceptance")
        plt.plot(global_accs.mean(0))
        plt.xlabel("iteration")

        plt.tight_layout()
        plt.show(block=False)

    def sampling_from_flow(
        self,
        n_sample,
        type_mc="MC",
        seed=0,
        QMC_engine=None,
        QMC_engine_args=None,
        inv_transform=True,
    ):
        if QMC_engine is None:
            QMC_engine = self.QMC_engine

        if type_mc == "MC":
            self.nf_sampler.rng_keys_nf = jax.random.PRNGKey(seed)
            sample = self.nf_sampler.sample_flow(n_sample)
            density_flow = self.nf_sampler.nf_model.log_prob(sample)
            density_target = self.target_vmap(sample)
            weights = jnp.exp(density_target - density_flow)
            return sample, weights
        elif type_mc == "QMC":
            QMC_sampler = QMC_engine(d=self.n_dim, seed=seed, **QMC_engine_args)
            model = self.nf_sampler.nf_model
            sample_qmc = jnp.asarray(
                qmc.MultivariateNormalQMC(
                    mean=np.zeros(self.n_dim),
                    seed=seed,
                    inv_transform=inv_transform,
                    engine=QMC_sampler,
                ).random(n_sample)
            )
            sample_qmc = model.inverse(sample_qmc)[0]
            sample_qmc = (
                sample_qmc * jnp.sqrt(jnp.diag(model.data_cov)) + model.data_mean
            )
            density_flow = model.log_prob(sample_qmc)
            density_target = self.target_vmap(sample_qmc)
            weights = jnp.exp(density_target - density_flow)
            return sample_qmc, weights
        else:
            raise ValueError("type_mc must be MC or QMC")

    def repeated_sampling(
        self,
        n_sample,
        n_rep,
        n_jobs=2,
        type_mc="MC",
        seed=0,
        QMC_engine=None,
        QMC_engine_args={},
        inv_transform=True,
    ):
        def loop(i):
            sample, weights = self.sampling_from_flow(
                n_sample,
                type_mc,
                seed=i + seed * 1000,
                QMC_engine=QMC_engine,
                QMC_engine_args=QMC_engine_args,
                inv_transform=inv_transform,
            )
            mean = jnp.average(a=sample, axis=0, weights=weights)
            if QMC_engine:
                qmc_name = QMC_engine.__name__ + "_" + str(inv_transform)
            else:
                qmc_name = ""
            return {f"mean_{i}": float(mean[i]) for i in range(n_dim)} | {
                "seed": i,
                "method": type_mc + qmc_name,
                "samples": sample,
                "weights": weights,
                "dim": self.n_dim
            }

        out = Parallel(n_jobs=n_jobs, backend="threading", verbose=10)(
            delayed(loop)(i) for i in range(n_rep)
        )
        df = pd.DataFrame(out)
        return df

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

    def make_box_plot(self, simulations):
        out = []
        for i in range(len(simulations)):
            df = self.repeated_sampling(**simulations[i])
            out.append(df)
        df_all = pd.concat(out)
        df_melt = pd.melt(
            df_all,
            id_vars=["seed", "method"],
            value_vars=[f"mean_{i}" for i in range(self.n_dim)],
        )
        sns.boxplot(x="variable", y="value", hue="method", data=df_melt)
        plt.show()
        return df_all, df_melt
    def make_simu(self,simulations):
        out = []
        for i in range(len(simulations)):
            df = self.repeated_sampling(**simulations[i])
            out.append(df)
        df_all = pd.concat(out)
        df_melt = pd.melt(
            df_all,
            id_vars=["seed", "method"],
            value_vars=[f"mean_{i}" for i in range(self.n_dim)],
        )
        #sns.boxplot(x="variable", y="value", hue="method", data=df_melt)
        #plt.show()
        return df_all, df_melt
    
    def repeated_sampling_dim(
        self,
        n_sample,
        n_rep,
        n_jobs=2,
        type_mc="MC",
        seed=0,
        QMC_engine=None,
        QMC_engine_args={},
        inv_transform=True,
    ):
        func = lambda x: jnp.sin(x[:, 0] * 10) * x[:, 0] ** 4
        def loop(i):
            sample, weights = self.sampling_from_flow(
                n_sample,
                type_mc,
                seed=i + seed * 1000,
                QMC_engine=QMC_engine,
                QMC_engine_args=QMC_engine_args,
                inv_transform=inv_transform,
            )
            mean = jnp.average(a=sample, axis=0, weights=weights)
            mean_func = jnp.average(a=func(sample), axis=0, weights=weights)
            if QMC_engine:
                qmc_name = QMC_engine.__name__ + "_" + str(inv_transform)
            else:
                qmc_name = ""
            return {f"mean_{i}": float(mean[i]) for i in range(n_dim)} | {
                "sin(x1)*x1^4": float(mean_func),
                "seed": i + seed * 1000,
                "method": type_mc + qmc_name,
                "dim": self.n_dim
            }

        out = Parallel(n_jobs=1, backend="threading", verbose=10)(
            delayed(loop)(i) for i in range(n_rep)
        )
        df = pd.DataFrame(out)
        return df
   
   
    def make_simu_dim(self,simulations):
        out = []
        for i in range(len(simulations)):
            df = self.repeated_sampling_dim(**simulations[i])
            out.append(df)
        df_all = pd.concat(out)
        df_melt = pd.melt(
            df_all,
            id_vars=["seed", "method"],
            value_vars=[f"mean_{i}" for i in range(self.n_dim)]+[f"sin(x1)*x1^4"],
        )
        return df_melt

    



# %%

n_dim = 2

model = MaskedCouplingRQSpline(n_dim, 6, [32,32], 8, jax.random.PRNGKey(10))
n_loop_training = 30


@jax.jit
def target_dualmoon(x, data=None):
    """
    Term 2 and 3 separate the distribution and smear it along the first and second dimension
    """
    term1 = 0.5 * ((jnp.linalg.norm(x) - 2) / 0.1) ** 2
    terms = []
    for i in range(n_dim):
        terms.append(-0.5 * ((x[i : i + 1] + jnp.array([-3.0, 3.0])) / 0.6) ** 2)
    return -(
        term1 - sum([logsumexp(i) for i in terms])
    )  
sampler = QMC_flow(
    n_dim,
    target_dualmoon,
    qmc.Sobol(d=n_dim),
    seed_training_flow=1,
    model=model,
    n_loop_training=n_loop_training,
)
sampler.trainning_flow()



n_size = 2**17
n_rep = 100
common_dict = {"n_sample": n_size, "n_rep": n_rep, "seed": 12, "n_jobs": 40}
simulations = [
    common_dict | {"type_mc": "MC"},
    common_dict | {"type_mc": "QMC", "QMC_engine": qmc.Sobol, "inv_transform": True},
  common_dict | {"type_mc": "QMC", "QMC_engine": qmc.Sobol, "inv_transform":False},
  common_dict | {"type_mc": "QMC", "QMC_engine": qmc.Halton, "inv_transform":True},
  common_dict | {"type_mc": "QMC", "QMC_engine": qmc.Halton, "inv_transform":False}]
plt.show()
plt.figure()
df_all,_= sampler.make_simu(simulations)
df_all.to_pickle("archi_test.zip")

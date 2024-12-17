import jax.numpy as jnp
import jax
import jaxopt
from jax import grad, hessian, jit, vmap, config
from jaxopt import ScipyBoundedMinimize
import jaxlib as jaxlib
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import copy
import matplotlib.pyplot as plt
import RegularizedRegression
from itertools import combinations, permutations

import FIVB_paper_figures
# import test_ord_regression

## points of Gauss-Hermite quadrature
n_GH = 20
zGH_i, wGH_i = np.polynomial.hermite.hermgauss(n_GH)

jax.config.update('jax_enable_x64', True)

def Qy_AC(z, alpha, delta):
    zz = z * delta + alpha
    sumexp = jnp.exp(jax.scipy.special.logsumexp(zz))

    return jnp.exp(zz)/sumexp

def CLM_model(z, c, der=0):
    bb = z[:, None] + c[None, :]
    if der == 0:
        val = np.diff(stats.norm.cdf(bb), axis=1)
        # for z>0 exploit the symmetry (to avoid division by zero)
        if any(z > 0):
            val[z > 0] = CLM_model(-z[z > 0], c, der)[:, ::-1]
    if der == 1:
        val = np.diff(stats.norm.pdf(bb), axis=1)
        # for z>0 exploit the symmetry (to avoid division by zero)
        if any(z > 0):
            val[z > 0] = -CLM_model(-z[z > 0], c, der)[:, ::-1]

    return val

def Qy_CL(z, c):
    ## calculate the distribution for a Cumulative Link model
    ## z: input value; if not a scalar (float) then vectorize using vmap
    ## c: finite interval limits, i.e., do not include -np.inf and np.inf
    CDF = "Gauss"
    # CDF = "logistic"
    # CDF = "Laplace"
    if len(z.shape) > 0:
        return vmap(Qy_CL, in_axes=[0, None])(z, c)
    zz = z + c
    if CDF == "Gauss":
        uu = jax.scipy.stats.norm.cdf(zz)
    elif CDF == "logistic":
        uu = jax.scipy.stats.logistic.cdf(zz)
    elif CDF == "Laplace":
        uu = jax.scipy.stats.laplace.cdf(zz)
    else:
        raise Exception("CDF undefined")

    A = jnp.eye(5, 6, 0) - jnp.eye(5, 6, 1)
    bb = jnp.eye(1, 6, 5)
    val = uu[None, :] @ A + bb[None, :]
    return val.ravel()

def Qy_hfa_AC(delta, alpha, eta, v):

    delta_A = jnp.array([[0, 0], [1,0], [0,1], [0,-1], [-1,0], [0,0]]) @ delta  # these are constraints on the coefficients
    delta_Ab = delta_A + jnp.array([2.0, 0, 0, 0, 0, -2.0])
    alpha_Ab = jnp.array([[0, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 0]]) @ alpha  # these are constraints on the coefficients
    qq_Ab = vmap(Qy_AC, in_axes=[0, None, None])(zGH_i * 2 * jnp.sqrt(v) + eta, alpha_Ab, delta_Ab)
    val = (wGH_i @ qq_Ab)/jnp.sqrt(jnp.pi)

    return val

def Qy_hfa(c, eta, v):
    PDP_prior = "Gauss"    ## in this case, gamma means the inverse of the variance
    # PDP_prior = "Laplace"   ## in this case, gamma means the inverse of the variance

    cc_Ab = jnp.array([[1, 0], [0, 1], [0, 0], [0, -1], [-1, 0]]) @ c  # these are constraints on the coefficients
    if PDP_prior == "Gauss":
        qq_Ab = vmap(Qy_CL, in_axes=[0, None])(zGH_i * 2 * jnp.sqrt(v), cc_Ab + eta)
        val = (wGH_i @ qq_Ab)/jnp.sqrt(jnp.pi)
    else:
        raise Exception("prior PDF undefined")

    return val

def NLL_AC(params, k_ntr, k_hfa, v):
    # negated log-likelihood
    LL_ntr = jnp.dot(k_ntr, jnp.log(Qy_hfa_AC(delta=params["delta"], alpha=params["alpha"], eta=0.0, v=v)))
    LL_hfa = jnp.dot(k_hfa, jnp.log(Qy_hfa_AC(delta=params["delta"], alpha=params["alpha"], eta=params["eta"], v=v)))
    return -(LL_ntr + LL_hfa)

def NLL(params, k_ntr, k_hfa, v):
    # negated log-likelihood
    LL_ntr = jnp.dot(k_ntr, jnp.log(Qy_hfa(c=params["c"], eta=0.0, v=v)))
    LL_hfa = jnp.dot(k_hfa, jnp.log(Qy_hfa(c=params["c"], eta=params["eta"], v=v)))
    return -(LL_ntr + LL_hfa)

def model_identification_AC(params_0, k_ntr, k_hfa, v_theta):

    solver = jaxopt.BFGS(fun=NLL_AC)
    res = solver.run(params_0, k_ntr=k_ntr, k_hfa=k_hfa, v=v_theta)

    return res.params

def model_identification(params_0, k_ntr, k_hfa, v_theta):

    solver = jaxopt.BFGS(fun=NLL)
    res = solver.run(params_0, k_ntr=k_ntr, k_hfa=k_hfa, v=v_theta)

    return res.params

def main():
    def jax2numCL(A):
        # transform the dictionary-indexed matrix (Hessian) into regular matrix
        A_out = np.zeros((3,3))
        A_out[0:2, 0:2] = A["c"]["c"]
        A_out[0:2, 2] = A["c"]["eta"]
        A_out[2, 0:2] = A["eta"]["c"]
        A_out[2, 2] = A["eta"]["eta"]
        return A_out

    def jax2numAC(A):
        # transform the dictionary-indexed matrix (Hessian) into regular matrix
        A_out = np.zeros((5,5))
        A_out[0:2, 0:2] = A["delta"]["delta"]
        A_out[0:2, 2:4] = A["delta"]["alpha"]
        A_out[0:2, 4] = A["delta"]["eta"]
        A_out[2:4, 0:2] = A["alpha"]["delta"]
        A_out[2:4, 2:4] = A["alpha"]["alpha"]
        A_out[2:4, 4] = A["alpha"]["eta"]
        A_out[4, 0:2] = A["eta"]["delta"]
        A_out[4, 2:4] = A["eta"]["alpha"]
        A_out[4, 4] = A["eta"]["eta"]

        return A_out

    save_DIR = "results/FIVB/"
    c_FIVB = np.array([-1.06, -0.394, 0, 0.394, 1.06])
    ## print general statistics
    if True:
        df, data, theta_FIVB = FIVB_paper_figures.import_FIVB_data(return_theta=True)
        X, y, u = data
        M, T = X.shape
        print("There are {} team, and {} games".format(M,T))
        k_hfa = df[df["HFA"] == 1]['Result'].value_counts().sort_index()
        k_ntr = df[df["HFA"] == 0]['Result'].value_counts().sort_index()
        k_ntr = 0.5 * (k_ntr.values + k_ntr.values[::-1])   # makes the neutral games symmetric
        T_ntr = k_ntr.sum()
        T_hfa = k_hfa.sum()
        k_hfa = k_hfa.values
        k_hfa_tilde = 0.5*(k_hfa + k_hfa[::-1])
        # k_hfa *= 0
        print("k_ntr:", k_ntr)
        print("k_hfa:", k_hfa)
        print("k_hfa_tilde:", k_hfa_tilde)
        c_FIVB = np.array([-1.06, -0.394, 0.0, 0.394, 1.06])
        c_hat_no_hfa = scipy.stats.norm.ppf((k_hfa_tilde+k_ntr).cumsum()/(k_hfa_tilde+k_ntr).sum())
        print("c_y from no-HFA assumption:", c_hat_no_hfa)

        H_ntr = FIVB_paper_figures.empirical_entropy(k_ntr)
        H_hfa = FIVB_paper_figures.empirical_entropy(k_hfa)
        H_all = (H_ntr * T_ntr + H_hfa * T_hfa)/(T_ntr + T_hfa)
        print("========\n log score when estimating from frequency: ")
        print("neutral={:.2f}, home={:.2f}, all={:.2f}".format(H_ntr, H_hfa, H_all))
        print("corresponding probabilities: ")
        print("neutral={:.2f}, home={:.2f}, all={:.2f}".format(np.exp(-H_ntr), np.exp(-H_hfa), np.exp(-H_all)))
        z = (df["Home_ranking"] - df["Away_ranking"]).values
        y = df["Result"].values
        scale = 125.0
        cc_FIVB = np.array([-1.06, -0.394, 0, 0.394, 1.06])
        Qy = Qy_CL(z / scale, cc_FIVB)
        val_FIVB = -np.log(Qy[np.arange(T), y])
        val_FIVB_all = val_FIVB.mean()
        val_FIVB_ntr = val_FIVB[df["HFA"].values == 0].mean()
        val_FIVB_hfa = val_FIVB[df["HFA"].values == 1].mean()

        print("=======")
        print("log-score for the FIVB algorithm")
        print("neutral={:.2f}, home={:.2f}, all={:.2f}".format(val_FIVB_ntr, val_FIVB_hfa, val_FIVB_all))
        print("corresponding proba for the FIVB algorithm")
        print("neutral={:.2f}, home={:.2f}, all={:.2f}".format(np.exp(-val_FIVB_ntr), np.exp(-val_FIVB_hfa), np.exp(-val_FIVB_all)))

        print("======")  ## participation per period
        cut_months = [6,12]
        m_ii = []
        g_ii = []
        date_start = pd.to_datetime('2001-1-01', format="%Y-%m-%d")
        for year in range(2021, 2023+1):
            for month in cut_months:
                date_end = pd.to_datetime(f'{year}-{month}-01', format="%Y-%m-%d")
                fi = (df["Date"].values >= date_start) & (df["Date"].values < date_end)
                date_start = date_end
                g_ii.append(fi.sum())   # number of games per period
                m_ii.append((np.abs(X[:,fi]).sum(axis=1)>0).sum())     # number of active teams
        print("periods (months):", cut_months)
        print("number of games:", g_ii)
        print("number of games:", m_ii)

        print("======")
        date_split = pd.to_datetime('2023-1-01', format="%Y-%m-%d")
        df_train = df[df["Date"] < date_split]
        df_val = df[df["Date"] >= date_split]
        countries_train = set(df_train["Home_team"]).union(df_train["Away_team"])
        countries_val = set(df_val["Home_team"]).union(df_val["Away_team"])
        countries_train_only = countries_train.difference(countries_val)
        countries_val_only = countries_val.difference(countries_train)
        print(f"{len(countries_train_only)} countries only before {date_split}:", countries_train_only)
        print(f"{len(countries_val_only)} countries only after {date_split}:", countries_val_only)

        print("======")
        first_game = np.zeros(M, int)
        last_game = np.zeros(M, int)
        number_of_games = np.zeros(M, int)
        for m in range(M):
            tt = np.argwhere(X[m,:] != 0 ).ravel()
            first_game[m] = tt[0]
            last_game[m] = tt[-1]
            number_of_games[m] = len(tt)
        countries = u["countries"]
        mi = first_game.argmax()
        print(f"the latest first game of {countries[mi]} on :", df.loc[first_game[mi], "Date"])
        mi = last_game.argmin()
        print(f"the earliest last game of {countries[mi]} on :", df.loc[last_game[mi], "Date"])
        ng = number_of_games.min()
        print(f"smallest number of games by {np.array(countries)[ng==number_of_games]}:", ng)
        ng = number_of_games.max()
        print(f"largest number of games by {np.array(countries)[ng==number_of_games]}:", ng)

    # identify CL/AC model from frequency and plot parameters
    if False:
        r_FIVB = [2.0, 1.5, 1.0]
        c_init = jnp.array([-1.06, -0.394]).astype(float)
        delta_init = jnp.array([1.0, 0.5]).astype(float)
        alpha_init = jnp.array([.0, .0]).astype(float)
        eta_init = 0.0
        v_vec = np.logspace(-3,0.2,4)

        params_out = []
        LL_out = np.zeros_like(v_vec)
        std_out = []
        Occam_factor = np.zeros_like(v_vec)

        params_out_AC = []
        LL_out_AC = np.zeros_like(v_vec)
        std_out_AC = []
        Occam_factor_AC = np.zeros_like(v_vec)

        params_out_tilde = []
        LL_out_tilde = np.zeros_like(v_vec)
        std_out_tilde = []
        Occam_factor_tilde = np.zeros_like(v_vec)
        params_0 = {"c": c_init, "eta": eta_init}
        params_0_AC = {"delta": delta_init, "alpha": alpha_init, "eta": eta_init}
        for k, v in enumerate(v_vec):
            res = model_identification(params_0, k_ntr, k_hfa, v)
            params_out.append(res)
            LL_out[k] = NLL(res, k_ntr=k_ntr, k_hfa=k_hfa, v=v)
            H = hessian(NLL)(res, k_ntr=k_ntr, k_hfa=k_hfa, v=v)
            V = np.linalg.inv(jax2numCL(H))
            std_out.append(np.sqrt(np.diag(V)))  # standard deviation
            Occam_factor[k] = 0.5 * np.log(np.linalg.det(2 * np.pi * V))
            # suppose eta=0
            res_tilde = model_identification(params_0, k_ntr, k_hfa_tilde, v)
            params_out_tilde.append(res_tilde)
            LL_out_tilde[k] = NLL(res_tilde, k_ntr=k_ntr, k_hfa=k_hfa_tilde, v=v)
            H = hessian(NLL)(res_tilde, k_ntr=k_ntr, k_hfa=k_hfa_tilde, v=v)
            Vc = np.linalg.inv(H["c"]["c"])
            std_out_tilde.append(np.sqrt(np.diag(Vc)))  # standard deviation
            Occam_factor_tilde[k] = 0.5 * np.log(np.linalg.det(2*np.pi*Vc))
            # AC model
            res_AC = model_identification_AC(params_0_AC, k_ntr, k_hfa, v)
            params_out_AC.append(res_AC)
            LL_out_AC[k] = NLL_AC(res_AC, k_ntr=k_ntr, k_hfa=k_hfa, v=v)
            H = hessian(NLL_AC)(res_AC, k_ntr=k_ntr, k_hfa=k_hfa, v=v)
            V = np.linalg.inv(jax2numAC(H))
            std_out_AC.append(np.sqrt(np.diag(V)))  # standard deviation
            Occam_factor_AC[k] = 0.5 * np.log(np.linalg.det(2 * np.pi * V))

        C = np.array([re["c"] for re in params_out])
        eta = np.array([re["eta"] for re in params_out])
        STD = np.array(std_out)
        C_tilde = np.array([re["c"] for re in params_out_tilde])
        eta_tilde = np.array([re["eta"] for re in params_out_tilde])
        STD_tilde = np.array(std_out)

        plt.rcParams.update({'font.size': 12})
        COLOR = ["g", "r", "b", "m", "k", "c"]
        (fig, ax) = plt.subplots()
        for i in range(2):
            LAB  = "$c_" + str(i) + "$"
            ax.plot(v_vec, C[:,i], linestyle="-", color=COLOR[i], label=LAB)
            ax.fill_between(v_vec, C[:, i] + 2.0*STD[:, i], C[:, i] - 2.0*STD[:, i], linestyle="--", color=COLOR[i], alpha=0.2)
            ax.plot(v_vec, C_tilde[:, i], linestyle="-.", color="k")
            # show the FIVB values
            ax.plot(v_vec, c_init[i] * np.ones_like(v_vec), ':', color=COLOR[i])
            # ax.fill_between(v_vec, C_tilde[:, i] + STD_tilde[:, i], C_tilde[:, i] - STD_tilde[:, i], linestyle="--", color=COLOR[i], alpha=0.2)
        i = 2
        ax.fill_between(v_vec, eta + 2.0*STD[:, i], eta - 2.0*STD[:, i], linestyle="--", color=COLOR[i], alpha=0.2)
        ax.plot(v_vec, eta, linestyle="-", color=COLOR[i], label="$\eta$")
        # ax.fill_between(v_vec, eta_tilde + STD_tilde[:, i], eta_tilde - STD_tilde[:, i], linestyle="--", color=COLOR[i], alpha=0.2)
        # ax.plot(v_vec, eta_tilde, linestyle="-", color="k")

        ax.set_xlim([0, 1.5])
        # ax.set_position([0.15, 0.15, 0.8, 0.8])
        ax.grid()
        ax.set_xlabel("$v_{\\theta}$")
        ax.set_ylabel("$\\hat{c}_{y}, \hat{\\eta}$",labelpad=0)
        ax.legend(loc='upper right')

        fig_dir = "/Users/leszek/Library/CloudStorage/Dropbox/Apps/Overleaf/Salma.internship/figures/"
        fig_name = "c.y_eta.png"
        # plt.savefig(fig_dir + fig_name, format="png")

        ### plot the numerical scores r_y
        (fig, ax) = plt.subplots()
        r_0 = 2.0
        c_0 = C[:, 0]
        for i in range(1, 3):
            LAB = "$r_" + str(i) + "$"
            if i == 2:
                c_y = np.zeros_like(v_vec)
            else:
                c_y = C[:, i]
            c_y1 = C[:, i-1]
            r_y = r_0 * scipy.stats.norm.cdf(c_0) * (scipy.stats.norm.pdf(c_y) - scipy.stats.norm.pdf(c_y1))
            r_y /= (scipy.stats.norm.pdf(c_0) * (scipy.stats.norm.cdf(c_y) - scipy.stats.norm.cdf(c_y1)))
            ax.plot(v_vec, r_y, linestyle="-", color=COLOR[i], label=LAB)
            # show the FIVB values
            ax.plot(v_vec, r_FIVB[i] * np.ones_like(v_vec), ':', color=COLOR[i])

        fig_name = "r.y.png"
        # plt.savefig(fig_dir + fig_name, format="png")

    ### calculate and save the cross-validation results
    if False:
        def calculate_loo(params, data, opt_scheduling=[], **kwargs):
            ##

            input_is_LIST = isinstance(params, list)
            if input_is_LIST:
                params_sample = params[0]   # one of the parameters on the list
            else:
                params_sample = params
            opt_vars = set(kwargs.keys())       # variables which we will scan during optimization
            if input_is_LIST and len(opt_vars) > 0:
                raise Exception("if input is a list, do not specify the search-over terms")
            if not opt_vars.issubset(set(params_sample.keys())):
                raise Exception("search-over terms should be in the variable(dictionary) 'params'")

            if input_is_LIST:
                params_in = copy.deepcopy(params)  # use input list
            else: # create a new list
                params_in = []
                L_vars = len(kwargs[list(opt_vars)[0]])
                for i in range(L_vars):
                    pp = copy.deepcopy(params)
                    for key in kwargs:
                        pp[key] = kwargs[key][i]     # all kwargs alements must have the same length=L_vars
                    params_in.append(copy.deepcopy(pp))

            L = len(params_in)      ## number of elements in the list
            ########
            if len(opt_scheduling) == 0:  # no optimization needed
                params_out = params_in
            else:
                params_out = RegularizedRegression.optimize_ALO(params_in, data, opt_scheduling)

            #  recalculate the values of ALO
            X, y, u = data
            loo = {"full": np.zeros(L),
                   "neutral": np.zeros(L),
                   "hfa": np.zeros(L)
                   }
            for k, pp in enumerate(params_out):
                loo["full"][k], loo_vec_k = RegularizedRegression.ALO(pp, data)
                loo["hfa"][k] = loo_vec_k[u["hfa"] == 1].mean()
                loo["neutral"][k] = loo_vec_k[u["hfa"] == 0].mean()

            return loo, params_out
            ######

        def line_search(values_of_elem, params, elem_in_params, position_in_element=0):
            ## calculate the ALO function along one dimension of elem_in_params at position position_in_element
            out = np.zeros_like(values_of_elem)
            params_tmp = copy.deepcopy(params)
            for k, v in enumerate(values_of_elem):
                params_tmp[elem_in_params][position_in_element] = v
                out[k], _ = RegularizedRegression.ALO(params_tmp, data)
            return out

        Split_Train_Data = False
        if Split_Train_Data:
            data_train, data_val = FIVB_paper_figures.import_FIVB_data(split=True)
            data = data_train
            X_train, y_train, u_train = data_train
            X_val, y_val, u_val = data_val
            train_ext = "_train"        # to add to the file name when saving
        else:
            df, data = FIVB_paper_figures.import_FIVB_data(split=False)
            train_ext = ""

        X, y, u = data
        M, T = X.shape

        params = {}
        params["WEIGHT=ARG"] = 0    # 0: "weight" is an externation weighting, 1: "weight" multiplies the argument inside the function
        params["scale"] = 1.0

        params["gamma"] = 1.0
        params["eta"] = 0.0
        Nc_tot = 5
        Nc = Nc_tot // 2
        ## the matrix Ac is used to map vector params["c"] (potentially reduced-rank) into a full vector with Nc_tot elements
        params["Ac"] = jnp.eye(Nc_tot, Nc, 0) - jnp.fliplr(jnp.eye(Nc_tot, Nc, Nc - Nc_tot))
        params["c"] = c_FIVB[:2].copy()

        Nr_tot = 6
        Nr = Nr_tot // 2
        params["Ar"] = jnp.eye(Nr_tot, Nr, 0) - jnp.fliplr(jnp.eye(Nr_tot, Nr, Nr - Nr_tot))
        params["r"] = np.array([2.0, 1.5, 1.0])

        Nr_cat = u["category"].max() + 1
        params["weight"] = np.ones(Nr_cat)  ## number of weights should be equal to the number of categories

        Ndelta_tot = 6
        Ndelta = Ndelta_tot // 2
        params["Adelta"] = jnp.eye(Ndelta_tot, Ndelta, 0) - jnp.fliplr(jnp.eye(Ndelta_tot, Ndelta, Ndelta - Ndelta_tot))
        params["delta"] = np.array([2.0, 1.5, 1.0])

        Nalpha_tot = 6
        Nalpha = Nalpha_tot // 2
        ## the matrix Aalpha is used to map vector params["a"] (potentially reduced-rank) into a full vector with Nalpha_tot elements
        params["Aalpha"] = jnp.eye(Nalpha_tot, Nalpha, 0) + jnp.fliplr(jnp.eye(Nalpha_tot, Nalpha, Nalpha - Nalpha_tot))
        params["alpha"] = np.array([0, 0, 0], float)

        gamma_vec = np.logspace(-2, 1, 10)
        ######################################################
        ##      logarithmic-loss
        params["LOSS_FUN"] = 0  ## use log-likelihood as loss function
        params["CDF"] = 0  ## use Gaussian CDF

        Compare_LOO_vs_ALO = False
        if Compare_LOO_vs_ALO:
            params["gamma"] = 0.5
            params["eta"] = 0.2
            alo = RegularizedRegression.ALO(params, data)
            alo_old = RegularizedRegression.ALO_old(params, data)
            loo = RegularizedRegression.LOO(params, data)

        if False:
            alo_out, params_out = calculate_loo(params, data, gamma=gamma_vec)
            save_file = "alo_logloss_eta0"
            np.savez(save_DIR + save_file + train_ext, alo=alo_out, params=params_out)

        ### optimize ALO,
        ## initialization & scheduling
        c_mask = np.ones(Nc, int)  ## all elements are to be optimized
        eta_mask = np.ones(1, int)  ## all elements are to be optimized
        weight_mask = np.ones_like(params["weight"], int)
        weight_mask[0] = 0

        ### optimize "c", keep "eta"=0
        if False:
            opt_scheduling = [{"c": c_mask}]       #(do not optimize eta)
            alo_out, params_out = calculate_loo(params, data, opt_scheduling=opt_scheduling, gamma=gamma_vec)
            save_file = "alo_logloss_opt_c"
            np.savez(save_DIR + save_file + train_ext, alo=alo_out, params=params_out)

        ### optimize "eta", keep "c^{FIVB}"
        if False:
            opt_scheduling = [{"eta": eta_mask}]  # (optimize eta)
            alo_out, params_out = calculate_loo(params, data, opt_scheduling=opt_scheduling, gamma=gamma_vec)
            save_file = "alo_logloss_opt_eta"
            np.savez(save_DIR + save_file + train_ext, alo=alo_out, params=params_out)

        ### optimize "c" and "eta"
        if False:
            opt_scheduling = [{"c": c_mask, "eta": eta_mask}]
            alo_out, params_out = calculate_loo(params, data, opt_scheduling=opt_scheduling, gamma=gamma_vec)
            save_file = "alo_logloss_opt_c_eta"
            np.savez(save_DIR + save_file + train_ext, alo=alo_out, params=params_out)

        ### optimize "weight" and "eta"
        if False:
            opt_scheduling = [{"weight": weight_mask, "eta": eta_mask}]
            alo_out, params_out = calculate_loo(params, data, opt_scheduling=opt_scheduling, gamma=gamma_vec)
            save_file = "alo_logloss_opt_weight_eta"
            np.savez(save_DIR + save_file + train_ext, alo=alo_out, params=params_out)

        ### evaluate FIVB-compliant "weight" and optimal "eta"=0.2
        if False:
            params["weight"] = np.array([1, 1.75, 2, 3.5, 4, 4.5, 5], float)
            params["eta"] = 0.2
            alo_out, params_out = calculate_loo(params, data, gamma=gamma_vec)
            save_file = "alo_logloss_FIVB_weight_eta_02"
            np.savez(save_DIR + save_file + train_ext, alo=alo_out, params=params_out)
        ############################################################
        ## FIVB cost-function
        params["LOSS_FUN"] = 1  ## use FIVB loss function
        params["CDF"] = 0  ## use Gaussian CDF
        params["weight"] = np.ones(Nr_cat)

        ### evaluate standard solution without optimization
        if False:
            alo_out, params_out = calculate_loo(params, data, gamma=gamma_vec)
            save_file = "alo_FIVB_eta0"
            np.savez(save_DIR + save_file + train_ext, alo=alo_out, params=params_out)

        ### evaluate standard solution with r_tilde(c^FIVB) and \eta=0.2
        if False:
            rr = FIVB_paper_figures.c2rtilde(params["c"] @ params["Ac"].T)
            params["r"] = rr[:3]
            params["eta"] = 0.2
            alo_out, params_out = calculate_loo(params, data, gamma=gamma_vec)
            save_file = "alo_FIVB_r_tilde_eta_02"
            np.savez(save_DIR + save_file + train_ext, alo=alo_out, params=params_out)

        ### optimize "r" keep "eta"=0
        r_mask = np.array([0, 1, 1], int)  # optimize two elements
        if False:
            params["LOSS_FUN"] = 1  ## use FIVB cost as loss function
            params["CDF"] = 0  ## use Gaussian CDF
            params["eta"] = 0.0
            r_mask = np.array([0, 1, 1], int)   # optimize two elements
            opt_scheduling = [{"r": r_mask}]
            alo_out, params_out = calculate_loo(params, data, opt_scheduling=opt_scheduling, gamma=gamma_vec)
            save_file = "alo_FIVB_opt_r_eta0"
            np.savez(save_DIR + save_file + train_ext, alo=alo_out, params=params_out)

        ### optimize "r" and "eta"
        if False:
            params["LOSS_FUN"] = 1  ## use FIVB cost as loss function
            params["CDF"] = 0  ## use Gaussian CDF
            r_mask = np.array([0, 1, 1], int)   # optimize r[1] and r[2], keep r[0]
            opt_scheduling = [{"r": r_mask, "eta": eta_mask}]
            alo_out, params_out = calculate_loo(params, data, opt_scheduling=opt_scheduling, gamma=gamma_vec)
            save_file = "alo_FIVB_opt_r_eta"
            np.savez(save_DIR + save_file + train_ext, alo=alo_out, params=params_out)

        ### for optimized "c", use r_tilde and eta=0.2
        if False:
            ## load optimized "c:
            save_file = "alo_logloss_opt_c_eta" + train_ext
            res = np.load(save_DIR + save_file + ".npz", allow_pickle=True)
            pp_opt_c = res["params"].tolist()

            params_list = copy.deepcopy(pp_opt_c)
            for pp in params_list:
                pp["LOSS_FUN"] = 1  ## use FIVB loss function
                pp["CDF"] = 0  ## use Gaussian CDF
                rr = FIVB_paper_figures.c2rtilde(pp["c"] @ pp["Ac"].T)
                pp["r"] = rr[:3]
                pp["eta"] = 0.2
            alo_out, params_out = calculate_loo(params_list, data)
            save_file = "alo_FIVB_opt_c_r_tilde_eta_02"
            np.savez(save_DIR + save_file + train_ext, alo=alo_out, params=params_out)

        ### optimize "r" and "eta" using optimized "c"
        if False:
            ## load optimized "c:
            save_file = "alo_logloss_opt_c_eta" + train_ext
            res = np.load(save_DIR + save_file +".npz", allow_pickle=True)
            pp_opt_c = res["params"].tolist()

            ## do the optimization of "r" and "eta
            params_list = copy.deepcopy(pp_opt_c)
            for pp in params_list:
                pp["LOSS_FUN"] = 1  ## use FIVB loss function
                pp["CDF"] = 0  ## use Gaussian CDF
            opt_scheduling = [{"r": r_mask, "eta": eta_mask}]
            alo_out, params_out = calculate_loo(params_list, data, opt_scheduling=opt_scheduling)
            save_file = "alo_FIVB_opt_c_r_eta"
            np.savez(save_DIR + save_file + train_ext, alo=alo_out, params=params_out)

        #######################################################
        ###  LOGISTIC CL model:
        ###  optimize "c" and "eta"
        #######################################################
        # show logistic model with values c_tilde, eta_tilde calculated from the ranking FIVB model (i,e probit model)
        if False:
            sigma = np.sqrt(np.pi/8)
            params["CDF"] = 1  ## use logistic CDF
            params["c"] /= sigma
            params["eta"] = 0
            params["scale"] = sigma
            alo_out, params_out = calculate_loo(params, data, gamma=gamma_vec)
            save_file = "alo_logloss_CL_logistic_tilde_FIVB_c_eta0"
            np.savez(save_DIR + save_file + train_ext, alo=alo_out, params=params_out)

        if False:
            ## load optimized "eta:
            save_file = "alo_logloss_opt_eta" + train_ext
            res = np.load(save_DIR + save_file + ".npz", allow_pickle=True)
            pp_opt_c = res["params"].tolist()

            ## do the optimization of "c" and "eta
            params_list = copy.deepcopy(pp_opt_c)
            sigma = np.sqrt(np.pi/8)
            for pp in params_list:
                pp["CDF"] = 1  ## use logistic CDF
                pp["c"] /= sigma
                pp["eta"] /= sigma
                pp["scale"] = sigma
            alo_out, params_out = calculate_loo(params_list, data)
            save_file = "alo_logloss_CL_logistic_tilde_FIVB_c_opt_eta"
            np.savez(save_DIR + save_file + train_ext, alo=alo_out, params=params_out)

        if False:
            ## load optimized "c:
            save_file = "alo_logloss_opt_c_eta" + train_ext
            res = np.load(save_DIR + save_file + ".npz", allow_pickle=True)
            pp_opt_c = res["params"].tolist()

            ## do the optimization of "c" and "eta
            params_list = copy.deepcopy(pp_opt_c)
            sigma = np.sqrt(np.pi/8)
            for pp in params_list:
                pp["CDF"] = 1  ## use logistic CDF
                pp["c"] /= sigma
                pp["eta"] /= sigma
                pp["scale"] = sigma
            alo_out, params_out = calculate_loo(params_list, data)
            save_file = "alo_logloss_CL_logistic_tilde_opt_c_eta"
            np.savez(save_DIR + save_file + train_ext, alo=alo_out, params=params_out)

        ## optimize c and eta for the logistic model
        if False:
            sigma = np.sqrt(np.pi / 8)
            params["LOSS_FUN"] = 0     ## log-score
            params["CDF"] = 1          ## logistic CDF
            params["scale"] = sigma
            opt_scheduling = [{"c": c_mask, "eta": eta_mask}]
            alo_out, params_out = calculate_loo(params, data, opt_scheduling=opt_scheduling, gamma=gamma_vec)
            save_file = "alo_logloss_CL_logistic_opt_c_eta"
            np.savez(save_DIR + save_file + train_ext, alo=alo_out, params=params_out)

        #######################################################
        ###  CL model with weights xi being part of the model
        ###  optimize "weight"
        #######################################################

        ### optimize "weight" and using optimized "c" and "eta"
        if False:
            ## load optimized "c:
            save_file = "alo_logloss_opt_c_eta" + train_ext
            res = np.load(save_DIR + save_file + ".npz", allow_pickle=True)
            pp_opt_c = res["params"].tolist()

            ## do the optimization of "weight"
            params_list = copy.deepcopy(pp_opt_c)
            for pp in params_list:
                pp["LOSS_FUN"] = 0  ## use loss function with weights as arguments
                pp["CDF"] = 0  ## use Gaussian CDF
                pp["scale"] = 1.0
                pp["WEIGHT=ARG"] = 1    ## weights are arguments
            opt_scheduling = [{"weight": weight_mask}]
            alo_out, params_out = calculate_loo(params_list, data, opt_scheduling=opt_scheduling)
            save_file = "alo_ell_weights_in_hat_c_eta_opt_weight"
            np.savez(save_DIR + save_file + train_ext, alo=alo_out, params=params_out)

        #######################################################
        ###  AC model
        ###  optimize "alpha" and "delta" and "eta"
        #######################################################
        alpha_mask = np.array([0, 1, 1], int)
        delta_mask = np.array([0, 1, 1], int)
        params["LOSS_FUN"] = 2  ## use AC log-likelihood as loss function
        if True:
            opt_scheduling = [{"alpha": alpha_mask, "eta": eta_mask}, {"delta": delta_mask}]
            alo_out, params_out = calculate_loo(params, data, opt_scheduling=opt_scheduling, gamma=gamma_vec)
            save_file = "alo_AC_opt_all"
            np.savez(save_DIR + save_file + train_ext, alo=alo_out, params=params_out)

        ## the 1-D visualization
        if False:
            kk = 4
            elem_name = "c"
            elem_k = 1
            elem_0 = pp_opt[kk][elem_name][elem_k]
            elem_in = elem_0 + np.linspace(-0.5, 0.3, 20)
            alo_out = line_search(elem_in,  pp_opt[kk], elem_name, elem_k)
            V_opt = np.linalg.inv(HH_opt[kk])
            k_HH = 1
            v_elem = V_opt[k_HH, k_HH]
            # v_elem = 1/HH_opt[kk][elem_name][elem_name][elem_k, elem_k]
            zz = (0.5 / v_elem) * (elem_in - elem_0) ** 2 + alo_opt[kk]
            fig, ax = plt.subplots()
            ax.plot(elem_in, alo_out)
        ####

    ### calculate and save the real-time adaptation
    if True:
        #################################################################
        def extract_optimal_params(save_file):
            ext = ".npz"
            res = np.load(save_DIR + save_file + ext, allow_pickle=True)
            alo = res["alo"].reshape((1,))[0]
            params_in = res["params"]
            idx = np.argmin(alo["full"])
            return params_in[idx]
        #################################################################
        def RT_ranking(steps_table, data, params_in, theta_0):
            pred_list = []
            theta_RT_list = []
            X, y, u = data
            params = copy.deepcopy(params_in)
            hfa = params["eta"] * u["hfa"]
            for k, step in enumerate(steps_table):
                params["update_step"] = step
                theta_RT, z_out = RegularizedRegression.SG_ranking(data, params, theta_init=theta_0)
                pred = vmap(RegularizedRegression.validation_fun, in_axes=[0, 0, None, 0, None])(z_out, y, 1.0, hfa,
                                                                                                 params)
                pred_list.append(pred)
                theta_RT_list.append(theta_RT)
            return pred_list, theta_RT_list
        #################################################################
        def calculation_prediction_metrics(pred_list):
            U = {}
            U["all"] = np.zeros(len(pred_list))
            U["hfa"] = np.zeros(len(pred_list))
            U["ntr"] = np.zeros(len(pred_list))
            return
            for k, pred in enumerate(pred_list):
                U["all"][k] = pred.mean()
                U["hfa"][k] = pred[ii_test_hfa].mean()
                U["ntr"][k] = pred[ii_test_ntr].mean()
            return U

        #################################################################

        save_DIR = "results/FIVB/"

        df, data, theta_FIVB = FIVB_paper_figures.import_FIVB_data(split=False, return_theta=True)
        theta_0 = theta_FIVB[:, 0]
        train_ext = ""
        X, y, u = data
        M, T = X.shape

        params = {}
        params.update(extract_optimal_params("alo_FIVB_eta0"))

        ## filtering parameters
        params_filter = copy.deepcopy(params)
        params_filter["scale"] = 125.0
        # params_filter["epsilon"] = 1.e-4
        xi_FIVB = np.array([1.0, 1.75, 2.0, 3.5, 4.0, 4.5, 5.0])

        # steps_table = np.logspace(-2, 0, 10)
        steps_table = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0])

        ## FIVB: xi_FIVB + eta =0, (original FIVB model)
        if True:
            params_filter["LOSS_FUN"] = 1  ## use FIVB loss function
            params_filter["eta"] = 0.0
            params_filter["weight"] = xi_FIVB
            params_filter["r"] = np.array([2, 1.5, 1])
            pred_list, theta_list = RT_ranking(steps_table, data, params_filter, theta_0)
            U = calculation_prediction_metrics(pred_list)
            save_file = "pred_FIVB"
            np.savez(save_DIR + save_file, pred=pred_list, U=U, theta=theta_list, steps=steps_table)
            print(save_file,":\n", U)

        ## FIVB: xi=xi_FIVB, eta=0.2
        if True:
            params_filter["LOSS_FUN"] = 1  ## use FIVB loss function
            params_filter["eta"] = 0.2
            params_filter["weight"] = xi_FIVB
            params_filter["r"] = np.array([2, 1.5, 1])
            pred_list, theta_list = RT_ranking(steps_table, data, params_filter, theta_0)
            U = calculation_prediction_metrics(pred_list)
            save_file = "pred_FIVB_xi_FIVB_eta_02"
            np.savez(save_DIR + save_file, pred=pred_list, U=U, theta=theta_list, steps=steps_table)
            print(save_file,":\n", U)

        ## FIVB: xi=xi_FIVB, eta=0.2, r_tilde
        if True:
            params_filter["LOSS_FUN"] = 1  ## use FIVB loss function
            params_filter["eta"] = 0.2
            params_filter["weight"] = xi_FIVB
            params_filter["r"] = np.array([2, 0.9, 0.25])
            pred_list, theta_list = RT_ranking(steps_table, data, params_filter, theta_0)
            U = calculation_prediction_metrics(pred_list)
            save_file = "pred_FIVB_xi_FIVB_eta_02_r_tilde"
            np.savez(save_DIR + save_file, pred=pred_list, U=U, theta=theta_list, steps=steps_table)
            print(save_file, ":\n", U)

        ## FIVB: xi=1 + eta=0.2
        if True:
            params_filter["LOSS_FUN"] = 1  ## use FIVB loss function
            params_filter["eta"] = 0.2
            params_filter["weight"] = np.ones_like(xi_FIVB)
            params_filter["r"] = np.array([2, 1.5, 1])
            pred_list, theta_list = RT_ranking(steps_table, data, params_filter, theta_0)
            U = calculation_prediction_metrics(pred_list)
            save_file = "pred_FIVB_xi_1_eta_02"
            np.savez(save_DIR + save_file, pred=pred_list, U=U, theta=theta_list, steps=steps_table)
            print(save_file,":\n", U)

        ## FIVB: xi=1 + eta=0.2 + r=[2, 0.9, 0.25]
        if True:
            params_filter["LOSS_FUN"] = 1  ## use log-score loss function
            params_filter["eta"] = 0.2
            params_filter["weight"] = np.ones_like(xi_FIVB)
            params_filter["r"] = np.array([2, 0.9, 0.25])
            pred_list, theta_list = RT_ranking(steps_table, data, params_filter, theta_0)
            U = calculation_prediction_metrics(pred_list)
            save_file = "pred_FIVB_xi_1_eta_02_r_tilde"
            np.savez(save_DIR + save_file, pred=pred_list, U=U, theta=theta_list, steps=steps_table)
            print(save_file, ":\n", U)

        ## FIVB_CL: xi=1 + eta=0.2
        if True:
            params_filter["LOSS_FUN"] = 0  ## use log-score loss function
            params_filter["eta"] = 0.2
            params_filter["weight"] = np.ones_like(xi_FIVB)
            pred_list, theta_list = RT_ranking(steps_table, data, params_filter, theta_0)
            U = calculation_prediction_metrics(pred_list)
            save_file = "pred_CL_xi_1_eta_02"
            np.savez(save_DIR + save_file, pred=pred_list, U=U, theta=theta_list, steps=steps_table)
            print(save_file, ":\n", U)

        ## FIVB_CL_logistic: xi=1 + eta=0.2
        if True:
            scale_new = 80.0  # approx =  126 * 0.62
            eta_new  = 0.12 #approx = 0.2/0.62
            params_filter["LOSS_FUN"] = 0  ## use log-score loss function
            params_filter["CDF"] = 1  ## use logistic CDF
            params_filter["eta"] = eta_new
            params_filter["scale"] = scale_new
            params_filter["weight"] = np.ones_like(xi_FIVB)
            pred_list, theta_list = RT_ranking(steps_table, data, params_filter, theta_0)
            U = calculation_prediction_metrics(pred_list)
            save_file = "pred_CL_logistic"
            np.savez(save_DIR + save_file, pred=pred_list, U=U, theta=theta_list, steps=steps_table)
            print(save_file, ":\n", U)
    None


if __name__ == "__main__":
    main()

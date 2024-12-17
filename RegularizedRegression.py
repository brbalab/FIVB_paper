import jax.numpy as jnp
import jax
import jaxopt
from jax import grad, hessian, jit, vmap, config
from functools import partial
from jaxopt import ScipyBoundedMinimize
import jaxlib as jaxlib
import numpy as np
import copy
import scipy
import FIVB_paper_figures

do_JIT = True
if not do_JIT:
    print("JIT is disabled")
    config.update("jax_disable_jit", True)

# class RegularizedRegression():
#     def __init__(self, X=None, y=None, u=None):
#         self._check_inputs(X, y)
#
#     ## verify the categories of the outputs
#     def _check_inputs(self, X, y, u=None):
#         # X : input/independent variables
#         # y : output/dependent
#         # u : auxiliary input variables
#         self.M, self.T = X.shape
#         if len(y) != self.T:
#             raise Exception("X should have as many rows as there are elements in y")
#         self.X = X
#         self.y = y
#         self.categories, counts = np.unique(y, return_counts=True)
#         self.L = len(self.categories)
#         self.categories_freq_ = counts/self.T
#         if self.L == 1:
#             raise Exception("nothing to fit: only one category in the data")
#
#         if u is not None:
#             if len(u) != self.T:
#                 raise Exception("u should have T elements")
#         else:  # define artificial classes = 0
#             u = np.zeros(self.T).astype(int)
#         self.u = u
#         self.categories_aux, counts = np.unique(self.u, return_counts=True)
#         self.categories_aux_freq_ = counts/self.T
#     def fit_params(self, params):
#         self.fit_parameters = params
#     def do_fit(self, model="AC+L2"):
#         data = (self.X, self.y, self.u)
#         ## if "weights" are not defined set them to 1
#         if "weight" not in self.fit_parameters:
#             self.fit_parameters.update({"weight": np.ones_like(self.categories_aux, float)})
#         pp = self.fit_parameters
#         if model == "AC+L2":
#             self.theta_hat = theta_hat(pp, data)
#         else:
#             raise Exception("model not defined")

def optimize_ALO(pp_init, data, opt_scheduling):
    @jit
    def ALO_partial(pp_var, pp_const, data):
        pp_combined = dict(**pp_var, **pp_const)
        return ALO(pp_combined, data)

    def dict2num(sch, gg, HH):
        # take the elements from the dictionaries gg and HH and construct vector and matrix
        # do not use the elements which are masked as defined in the dictionary sch
        KK = 0  # size of the vector to be optimized
        for elem in sch:
            KK += np.sum(sch[elem])
        grad_ALO = np.zeros((KK,))
        Hess_ALO = np.zeros((KK, KK))
        k_start = 0
        for elem in sch:
            bb = (sch[elem] == np.ones_like(sch[elem]))  ## for boolean indexing
            # k_end = k_start + sch[elem].sum()
            k_end = k_start + (bb * 1).sum()
            # extract gradient elements
            grad_ALO[k_start: k_end] = gg[elem].reshape(len(bb),)[bb]   ## take only elements index with True
            k_start2 = 0
            for elem2 in sch:
                bb2 = (sch[elem2] == np.ones_like(sch[elem2]))  ## for boolean indexing
                # k_end2 = k_start2 + sch[elem2].sum()
                k_end2 = k_start2 + (bb2 * 1).sum()
                # extract Hessian elements
                Hess_ALO[k_start: k_end, k_start2: k_end2] = HH[elem][elem2].reshape((len(bb), len(bb2)))[bb][:, bb2]
                k_start2 = k_end2
            k_start = k_end

        return grad_ALO, Hess_ALO

    input_is_LIST = isinstance(pp_init, list)
    if not input_is_LIST:
        pp_init = [pp_init]  # make it a list

    step = 0.5
    maxrounds = 10
    pp_out_list = []
    for pp_init_k in pp_init:
        pp_out = copy.deepcopy(pp_init_k)

        # replace scalars by 1-D np.arrays (to faciliate mapping between pytrees and vectors
        # for key in pp_out:
        #     if isinstance(pp_out[key], float):
        #         pp_out[key] = np.array(pp_out[key]).reshape(1,)
        J_ref = np.inf
        eps_precision = 1e-6
        for i_round in range(maxrounds):
            # verification if there is improvement
            J_ref_old = J_ref
            J_ref = ALO(pp_out, data)
            HAS_AUX = isinstance(J_ref, tuple)
            if HAS_AUX:
                J_ref = J_ref[0]
            if np.abs(J_ref - J_ref_old) < eps_precision * J_ref:
                break
            # optimize using the order defined in the list opt_scheduling
            for sch in opt_scheduling:
                ## split the dictionaries
                pp_var = {key: pp_out[key] for key in sch}
                pp_const = {key: pp_out[key] for key in set(pp_out.keys()).difference(pp_var.keys())}
                ## calculate gradient and Hessian, the function hessian() does not work here
                gg = grad(ALO_partial, has_aux=HAS_AUX)(pp_var, pp_const, data)
                HH = jax.jacrev(jax.jacrev(ALO_partial, has_aux=HAS_AUX), has_aux=HAS_AUX)(pp_var, pp_const, data)
                if HAS_AUX:
                    gg = gg[0]
                    HH = HH[0]

                ### Here we transform the gradient and the Hessian jax variables into the vector/matrix forms
                grad_ALO, Hess_ALO = dict2num(sch=sch, gg=gg, HH=HH)

                ####  Now, we can execute one step of the Newton (or steepest descent) method
                det_Hess = np.linalg.det(Hess_ALO)
                if det_Hess > 0:
                    ## solve linear equation (for a Hermitian matrix)
                    d_pp = jax.scipy.linalg.solve(Hess_ALO, grad_ALO, assume_a="her")
                else:
                    ## steepest descent
                    print("Hessian not invertible: using gradient")
                    d_pp = grad_ALO * step
                ## update the parameters in pp_out
                k_start = 0
                for elem in sch:
                    bb = (sch[elem] == np.ones_like(sch[elem]))  ## for boolean indexing
                    L_bb = (bb * 1).sum()
                    k_end = k_start + L_bb
                    if L_bb==1:  ## this is a scalar
                        pp_out[elem] -= d_pp[k_start]
                    else:
                        pp_out[elem][bb] -= d_pp[k_start: k_end]    ## update elements
                    k_start = k_end

        pp_out_list.append(copy.deepcopy(pp_out))
    if input_is_LIST:
        pp_out = pp_out_list
    else:
        pp_out = pp_out_list[0]
    return pp_out


@jit
def ALO(pp, data):
    X, y, u = data
    xi = pp["weight"][u["category"]]
    hfa = u["hfa"] * pp["eta"]
    M, T = X.shape

    theta_hat_out = theta_hat(pp, data)
    H = hessian(J_obj)(theta_hat_out, data, pp)
    H_inv = jax.scipy.linalg.inv(H)
    a = jnp.sum(jnp.multiply(H_inv @ X, X), axis=0)

    z_hat = X.T @ theta_hat_out
    # z_hat_hfa = z_hat + hfa
    ######

    # ell_dot = vmap(grad(loss_fun), in_axes=[0, 0, 0, 0, None])(z_hat_hfa, y, xi, hfa, pp)
    # ell_ddot = vmap(hessian(loss_fun), in_axes=[0, 0, 0, 0, None])(z_hat_hfa, y, xi, hfa, pp)
    ell_dot = vmap(grad(loss_fun), in_axes=[0, 0, 0, 0, None])(z_hat, y, xi, hfa, pp)
    ell_ddot = vmap(hessian(loss_fun), in_axes=[0, 0, 0, 0, None])(z_hat, y, xi, hfa, pp)

    # ell_dot2 = jnp.array([grad(loss_fun)(z_hat[i], y[i], xi[i], pp) for i in range(T)])
    # ell_ddot2 = jnp.array([hessian(loss_fun)(z_hat[i], y[i], xi[i], pp) for i in range(T)])

    z_hat_approx = z_hat + ell_dot * a /(1 - ell_ddot * a)
    # z_hat_approx_hfa = z_hat_approx + hfa
    # pred = vmap(validation_fun, in_axes=[0, 0, 0, 0, None])(z_hat_approx_hfa, y, xi, hfa, pp)
    pred = vmap(validation_fun, in_axes=[0, 0, 0, 0, None])(z_hat_approx, y, xi, hfa, pp)

    return jnp.sum(pred)/len(y), pred

def GALO(pp, data, leave_out):
    ## generalized ALO, uses set-of-sets ii to indicate which elements are left-out
    ## becomes ALO when ii=[[0], [1], [2], ... , [T-1]]
    X, y, u = data
    xi = pp["weight"][u["category"]]
    theta_hat_out = theta_hat(pp, data)
    H = hessian(J_obj)(theta_hat_out, X, y, u, pp)
    H_inv = jax.scipy.linalg.inv(H)
    # A = [X[:, ii].T @ H_inv @ X[:, ii]  for ii in leave_out]

    z_hat = X.T @ theta_hat_out + pp["eta"] * u["hfa"]
    ell_dot = vmap(grad(loss_fun), in_axes=[0, 0, 0, None])(z_hat, y, xi, pp)
    ell_ddot = vmap(hessian(loss_fun), in_axes=[0, 0, 0, None])(z_hat, y, xi, pp)

    pred = 0.0
    for ii in leave_out:
        A = X[:, ii].T @ H_inv @ X[:, ii]
        z_hat_approx = z_hat[ii] + A @ jnp.linalg.inv(jnp.eye(len(ii)) - jnp.diag(ell_ddot[ii]) @ A ) @ ell_dot[ii]
        pred += vmap(validation_fun, in_axes=[0, 0, 0, None])(z_hat_approx, y[ii], xi[ii], pp).mean()
    return jnp.sum(pred)/len(leave_out)

def LOO(pp, data):
    X, y, u = data
    xi = pp["weight"][u["category"]]
    hfa = u["hfa"] * pp["eta"]
    M, T = X.shape

    pred = jnp.zeros(T)
    tt_full = jnp.arange(T)
    t_keep = tt_full
    for t in range(T):
        x_test = X[:, t]
        t_keep = jnp.delete(tt_full, t)

        theta_hat_out = theta_hat(pp, data, t_keep)
        z_hat = jnp.dot(x_test, theta_hat_out)
        pred_result = validation_fun(z_hat, y[t], xi[t], hfa[t], pp)
        pred = pred.at[t].set(pred_result)

    return jnp.mean(pred), pred

def LOO_explicit(pp, data):
    X, y, u = data
    hfa = u["hfa"]
    M, T = X.shape

    pred = jnp.zeros(T)
    u_train = {f : 0 for f in u}  # initialize the dictionary
    for t in range(T):
        X_train = jnp.delete(X, t, axis=1)
        y_train = jnp.delete(y, t)
        for f in u_train:
            u_train[f] = jnp.delete(u[f], t)

        x_test = X[:, t]
        theta_hat_out = theta_hat(pp, (X_train, y_train, u_train))
        z_hat = jnp.dot(x_test, theta_hat_out) + pp["eta"] * hfa[t]
        pred_result = validation_fun(z_hat, y[t], 1.0, pp)
        pred = pred.at[t].set(pred_result)

    return jnp.mean(pred), pred

@jit
def theta_hat(pp, data, t_keep=None):
    maxiter = 200
    X, _, _ = data
    M, T = X.shape

    solver = jaxopt.BFGS(fun=J_obj, maxiter=maxiter, implicit_diff=True, verbose=False)
    theta_init = jnp.zeros(M)
    res = solver.run(theta_init, data, pp, t_keep)
    return res.params

@jit
def J_obj(theta, data, pp, t_keep=None):
    X, y, u = data
    xi = pp["weight"][u["category"]]
    hfa = u["hfa"] * pp["eta"]

    z = X.T @ theta
    # z += pp["eta"] * hfa
    loss = vmap(loss_fun, in_axes=[0, 0, 0, 0, None])(z, y, xi, hfa, pp)      # vectorization
    return jnp.sum(loss[t_keep]) + regularization_fun(theta, pp)

@jit
def regularization_fun(theta, params):
    REGULARIZATION_FUN = "L2"
    if REGULARIZATION_FUN == "L2":
        # this is the ridge regularization function
        gamma = params["gamma"]
        return jnp.sum(0.5 * gamma * theta**2)
    elif REGULARIZATION_FUN == "XXXX":
        None
    else:
        raise Exception("regularization function undefined")

@jit
def validation_fun(z, y, xi, hfa, params):
    # validataion function, uses the elementary loss function
    scale = params["scale"]

    # z = jax.lax.select(params["WEIGHT=ARG"] == 1, jnp.dot(xi, z), z)  ## xi as arguments (when "weight=arg"=1) or not
    z = z / scale + hfa  ## scaling and offset
    # functions on the list loss_functions_list are selected using the value of fun_switch["LOSS_FUN"]
    loss_functions_list = [logarithmic_loss_CL, logarithmic_loss_CL]#, logarithmic_loss_AC]
    ell = jax.lax.switch(params["LOSS_FUN"], loss_functions_list, z, y, params)

    return ell

@jit
def loss_fun(z, y, xi, hfa, params):
    # loss function, uses
    # all changes to the loss function should be done here (such as adding the offset, weights, etc
    # this is the scalar ordinal model (should not contain vector arguments when called; use vmap)
    scale = params["scale"]

    # z = jax.lax.select(params["WEIGHT=ARG"] == 1, jnp.dot(xi, z), z)  ## xi as arguments (when "weight=arg"=1) or not
    z = z / scale + hfa  ## scaling and offset
    # functions on the list loss_functions_list are selected using the value of fun_switch["LOSS_FUN"]
    loss_functions_list = [logarithmic_loss_CL, fivb_loss]  # , logarithmic_loss_AC]
    ell = jax.lax.switch(params["LOSS_FUN"], loss_functions_list,z, y, params)
    # ell_out = jax.lax.select(params["WEIGHT=ARG"] == 1, ell, jnp.dot(xi, ell))  ## xi as weights (when "weight=arg"=0) or not
    ell_out = jnp.dot(xi, ell)
    return ell_out

@jit
def fivb_loss(z, y, params):
    rr = params["Ar"] @ params["r"]
    cc = params["Ac"] @ params["c"]

    dr = - jnp.diff(rr)
    zz = z + cc
    psi = jax.scipy.stats.norm.cdf(zz) * zz + jax.scipy.stats.norm.pdf(zz)
    ell = jnp.dot(psi, dr) + jnp.dot(rr[-1] - rr[y], z)

    return ell

@jit
def logarithmic_loss_AC(z, y, params):
    ## this is the scalar ordinal model (should not be vectorized after being called)
    alpha = params["Aalpha"] @ params["alpha"]
    delta = params["Adelta"] @ params["delta"]

    v = alpha + delta * z
    # negated log-loss
    ell = jax.scipy.special.logsumexp(v) - v[y]

    return ell

@jit
def logarithmic_loss_CL(z, y, params):
    ## this is the scalar ordinal model (should not be vectorized after being called)
    cc = params["Ac"] @ params["c"]
    Nc_tot = len(cc)

    is_positive = (z > 0) * 1
    v = z * (1 - 2 * is_positive) + cc

    ell_tmp_log = ell_CL_log(v, params)
    ## we are exploiting the fact that ell_y(z) = ell_{L-1-y}(-z)
    ## this should be generalized for arbitrary c !!!!
    ell = ell_tmp_log[y] * (1 - is_positive) + ell_tmp_log[Nc_tot - y] * is_positive

    return ell

@jit
def ell_CL_log(zz, params):
    ## implement ell_CL using logarithms
    ## log(cdf(a)-cdf(b)) = logcdf(a) + log[1-exp(logcdf(a)-logcdf(b))]
    ## since z[-1]=-inf, and z[L] = inf, we have: logcdf(z[-1]) = -inf and logcdf(z[L]) = 0

    # params["CDF"]=0:  CDF = "Gauss"
    # params["CDF"]=1: CDF = "logistic"

    uu = jax.lax.select(params["CDF"] == 0, jax.scipy.stats.norm.logcdf(zz), jnp.log(jax.scipy.stats.logistic.cdf(zz)))

    Nc_tot = len(uu)
    A1 = jnp.eye(Nc_tot, Nc_tot + 1, 0)
    a = uu @ A1

    A2 = -jnp.eye(Nc_tot, Nc_tot , 0) + jnp.eye(Nc_tot, Nc_tot, -1)
    expd = jnp.exp(-uu @ A2)

    A3 = jnp.eye(Nc_tot, Nc_tot + 1, 1)
    b = jnp.log(1 - expd @ A3)

    return - (a + b)


####################################################################################################
def SG_ranking(data, params, theta_init=None):
    X, y, u = data
    M, T = X.shape  # M: number of players, T: number of games

    ## initialize the skills
    if theta_init is None:
        theta = jnp.zeros(M)  # all-zeros
    elif isinstance(theta_init, float) or isinstance(theta_init, int):
        theta = jnp.ones(M) * theta_init  # all equal values
    else:
        theta = theta_init.copy()  # predefined values

    theta_all = np.zeros((M, T))  # all skills through time
    z_out = np.zeros(T)  # filtering result through time

    scale = 1.0 if "scale" not in params else params["scale"]
    eta = 0.0 if "eta" not in params else params["eta"]

    xi = params["weight"][u["category"]]
    hfa = eta * u["hfa"]

    update_step = params["update_step"]

    ####   main loop
    for t in range(T):
        x_t = X[:, t]
        z_t = jnp.dot(x_t, theta)
        # z_out = z_out.at[t].set(jnp.dot(x_t, theta))
        z_out[t] = z_t
        y_t = y[t]

        ########################################################################
        ## this is where we depend on the model
        ##
        grad_t = grad(loss_fun)(z_t, y_t, xi[t], hfa[t], params)
        ##
        ########################################################################
        do_test = True
        print_test = True
        if do_test:
            c_FIVB = params["Ac"] @ params["c"]
            r_FIVB = params["Ar"] @ params["r"]
            Qy = FIVB_paper_figures.FIVB_calculation.Qy_CL(z_t / scale, c_FIVB)
            Expected_score = Qy @ r_FIVB
            grad_FIVB = Expected_score - r_FIVB[y_t]
            delta = grad_FIVB * update_step * xi[t]
            if print_test:
                print("==========")
                print("t=",t)
                print("z_t=",z_t)
                print("Qy=",Qy)
                print("Expected_score=", Expected_score)
                print("grad_FIVB = Expected_score - r^FIVB_y=", grad_FIVB)
                print("y[t]=",y_t)
                print("xi[t]=", xi[t])
                print("scale * grad_FIVB * update_step * xi[t]=", scale * delta)
                fi = np.where(x_t!=0)
                print("theta[",fi, "]before change:", theta[fi])

        theta = theta - update_step * (scale**2) * x_t * grad_t  ## we need square
                                                    # of the scale because it is included in the gradient calculation
        print("theta[",fi, "] after change:", theta[fi])
        theta_all[:, t] = theta
        # theta_all = theta_all.at[:, t].set(theta)

    return theta_all, z_out


def real_time_ranking(data, params, theta_init=None, v0=None):

    X, y, u = data
    M, T = X.shape  # M: number of players, T: number of games

    ## initialize the skills
    if theta_init is None:
        theta = jnp.zeros(M)        # all-zeros
    elif isinstance(theta_init, float) or isinstance(theta_init, int):
        theta = jnp.ones(M) * theta_init  # all equal values
    else:
        theta = theta_init.copy()       # predefined values  (suppose vector for vSKF, and matrix for KF)

    theta_all = np.zeros((M, T))  # all skills through time
    z_out = np.zeros(T)     # filtering result through time

    xi = params["weight"][u["category"]]
    hfa_ind = u["hfa"]

    scale = 1.0 if "scale" not in params else params["scale"]
    eta = 0.0 if "eta" not in params else params["eta"]
    rating_algorithm = params["rating_algorithm"]
    
    delta_time = jnp.zeros(T)  # time does not matter (in SG)

    if rating_algorithm in {"KF", "vSKF"}:
        beta = 1.0 if "beta" not in params else params["beta"]
        epsilon = params["epsilon"]
        delta_time = jnp.concatenate([jnp.array(0.0), jnp.diff(u["time_stamp"])])
        if isinstance(v0, float) or isinstance(v0, int):
            V_t = jnp.eye(M) * v0 if rating_algorithm == "KF" else np.ones(M) * v0  # covariance matrix or vector from scalar
    
    if rating_algorithm == "SG":
        update_step = params["update_step"]

    V_out = [] # covariance matrices/vectors/scalars

    # remove key which are strings from the dictionary to allow jit-ing the function
    params_wo_string = {key: value for key, value in params.items() if not isinstance(value, str)}
    beta_t = 1.0
    ####   main loop
    for t in range(T):
        if rating_algorithm in {"KF", "vSKF"}:
            beta_t = beta ** delta_time[t]  # time-dependent version of beta
            epsilon_t = epsilon * delta_time[t]

        x_t = X[:, t]
        z_out[t] = jnp.dot(x_t, theta)
        z_t = beta_t * jnp.dot(x_t, theta)
        y_t = y[t]

        ########################################################################
        ## this is where we depend on the model
        ##
        grad_t = grad(loss_fun)(z_t / scale + eta * hfa_ind[t], y_t, xi[t], params_wo_string)
        hess_t = hessian(loss_fun)(z_t / scale + eta * hfa_ind[t], y_t, xi[t], params_wo_string)
        ##
        ########################################################################

        # prepare update of the variance
        if rating_algorithm == "KF":
            V_bar = (beta_t ** 2) * V_t + epsilon_t * jnp.identity(M)  # matrix
            Vx_t = V_bar @ x_t
            omega_t = jnp.dot(x_t, Vx_t)
            V_t = V_bar - jnp.outer(Vx_t, Vx_t * (hess_t / (scale ** 2 + hess_t * omega_t)))
            # update skills
            theta = beta_t * theta - Vx_t * (scale * grad_t) / (scale ** 2 + hess_t * omega_t)
            V_out.append(V_t.copy())

        elif rating_algorithm == "vSKF":
            V_bar = (beta_t ** 2) * V_t + epsilon_t  # vector
            Vx_t = V_bar * x_t      ## point-wise multiplication
            omega_t = jnp.dot(x_t, Vx_t)
            V_t = V_bar * (1 - V_bar * jnp.abs(x_t) * (hess_t / (scale ** 2 + hess_t * omega_t)))
            # update skills
            theta = beta_t * theta - Vx_t * (scale * grad_t) / (scale ** 2 + hess_t * omega_t)
            V_out.append(V_t.copy())

        elif rating_algorithm == "SG":
            theta = theta - update_step * x_t * grad_t / scale

        theta_all[:, t] = theta

    return theta_all, z_out, V_out

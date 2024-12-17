import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.stats as stats
import pandas as pd
import os
import FIVB_calculation
import RegularizedRegression
from jax import vmap, grad

# import test_ord_regression
import dictionaries

plt.rcParams.update({
    "text.usetex": True,
    'font.size': 12,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",  # Use a serif font
    "font.serif": ["Computer Modern"]  # Use Computer Modern for the serif font
})
# plt.rcParams.update({'font.size': 12})

def merge_FIVB_files():
    root_dir = "FIVB/"
    df_new = pd.read_csv(root_dir + "FIVB_data.csv")
    df_old = pd.read_csv(root_dir + "FIVB_data_time_machine.csv")
    df = pd.concat([df_new, df_old], ignore_index=True)
    df = df.drop_duplicates()
    df.rename(columns={'dates': 'Date',
                       'home_team': 'Home_team',
                       'away_team': 'Away_team',
                       'home_worldranking': 'Home_ranking',
                       'away_worldranking': 'Away_ranking',
                       'home_results': 'Home_result',
                       'away_results': 'Away_result',
                       'increment': 'Increment',
                       }, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df.sort_values(by='Date', inplace=True, ignore_index=True)
    return df

def merge_wikipedia_files():
    root_dir = "FIVB/Wikipedia_files"
    # Traverse the directory tree
    k = 0
    df_out = None
    for root, dirs, files in os.walk(root_dir):
        year_str = root.split("/")[-1]
        # if year_str!="2021":
        #     continue
        for file in files:
            if file.endswith(".csv"):
                k += 1
                # Construct the full file path
                file_path = os.path.join(root, file)
                year_str = root.split("/")[-1]
                # Read the file
                df = pd.read_csv(file_path)
                df["Date"] = df["Date"].str.replace("*", "")     ## wrong character
                try:
                    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
                except:
                    try:
                        df["Date"] = pd.to_datetime(df["Date"], format="%d %B %Y")
                    except:
                        try:
                            df["Date"] = pd.to_datetime(df["Date"] + " " + year_str, format="%d %b %Y")
                        except:
                            print("could not convert the date!!!!!!")
                cols = list(df.columns)
                score_index = cols.index("Score")
                df.rename(columns={cols[score_index-1]: "Home_team", cols[score_index+1]: "Away_team"},
                          inplace=True)
                if df_out is None:
                    df_out = df
                else:
                    df_out = pd.concat([df_out, df], ignore_index=True)

    print("imported {} files".format(k))
    df_out.sort_values(by='Date', inplace=True, ignore_index=True)

    df_out["Home_result"] = df_out["Score"].str[0].astype(int)
    df_out["Away_result"] = df_out["Score"].str[2].astype(int)

    return df_out

def import_FIVB_data(print_values=False, split=False, return_theta=False):
    # define the training/validation split
    if split:
        split_date = pd.to_datetime("2023-01-01", format="%Y-%m-%d")  # training/validation split date

    try:
        df = pd.read_csv("FIVB/FIVB_all.csv")
        print("reading file from disk")
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")

    except:
        print("merging files from scratch")

        df_wiki = merge_wikipedia_files()
        df_FIVB = merge_FIVB_files()

        cut_off_date = pd.to_datetime("2021-01-01", format="%Y-%m-%d")  # cut-off date
        df_wiki = df_wiki[df_wiki["Date"] >= cut_off_date]
        df_FIVB = df_FIVB[df_FIVB["Date"] >= cut_off_date]


        # use the coutry names which are in the predefined dictionary
        c2c = {"United States": "USA", "DR Congo": "Congo DR",
               "South Korea": "Korea Republic", "Iran": "IR Iran",
               "China": "China PR", "Kyrgyzstan": "Kyrgyz Republic",
               "AChina": "China PR", "GermanyB": "Germany",
               "Morocco[a]": "Morocco"}
        df_wiki.replace(c2c, inplace=True)
        a2c = dictionaries.abr2country_FIVB  ### import abreviation-to-country_name dictionary (used by FIVB)
        c2a = {a2c[i]: i for i in a2c}      ## create country-to-abbreviation dictionary
        df_wiki.replace(c2a, inplace=True)

        ## remove invalid games
        df_FIVB = df_FIVB[df_FIVB["Increment"] != 0.01]
        df_FIVB = df_FIVB[df_FIVB["Increment"] != 0.0]

        countries = set(df_FIVB["Home_team"]).union(df_FIVB["Away_team"])
        not_countries = countries.difference(set(a2c))  ## names which are not countries
        if len(not_countries)>0:
            raise Exception("Some countries in the df_FIVB are not on the list")

        # df_wiki = df_wiki[~df_wiki["Home_team"].isin(not_countries) & ~df_wiki["Away_team"].isin(not_countries)]

        ## reset indexing
        df_wiki.reset_index(drop=True, inplace=True)
        df_FIVB.reset_index(drop=True, inplace=True)
        ## iterative merge of venues
        df_no_venue_update = pd.read_csv("FIVB/no_venue_FIVB_games_updated.csv")
        df_FIVB["Venue"] = ""
        for i in range(len(df_FIVB)):
            Date = df_FIVB.iloc[i]["Date"]
            Home_team, Away_team = df_FIVB.iloc[i]["Home_team"], df_FIVB.iloc[i]["Away_team"]
            Home_result, Away_result = df_FIVB.iloc[i]["Home_result"], df_FIVB.iloc[i]["Away_result"]
            match = ((df_wiki["Home_team"] == Home_team) & (df_wiki["Away_team"] == Away_team)
                     & (df_wiki["Home_result"] == Home_result) & (df_wiki["Away_result"] == Away_result)
                     )
            match2 = ((df_wiki["Home_team"] == Away_team) & (df_wiki["Away_team"]==Home_team)
                      & (df_wiki["Home_result"] == Away_result) & (df_wiki["Away_result"] == Home_result)
                      )
            match_date = df_wiki["Date"] == Date
            # match_date = np.abs((df_wiki["Date"] - Date).values / np.timedelta64(1, 'D')) <= 1
            df_match = df_wiki[(match | match2) & match_date]

            if len(df_match) == 1:
                df_FIVB.loc[i, "Venue"] = df_match.iloc[0]["Venue"]
            elif len(df_match) > 1:
                print("too many matches")
            elif len(df_match) == 0:  # no check the hand-crafted file with venues
                match = ((df_no_venue_update["Home_team"] == Home_team) & (df_no_venue_update["Away_team"] == Away_team)
                         & (df_no_venue_update["Home_result"] == Home_result) & (df_no_venue_update["Away_result"] == Away_result)
                         )
                if match.sum() > 0:
                    df_match = df_no_venue_update[match]
                    df_FIVB.loc[i, "Venue"] = df_match.iloc[0]["Venue"]

        ## manualy set the venues
        ## eliminate games Denmark forfeited in January 2021
        date_cond = df_FIVB["Date"] < pd.to_datetime("2021-01-17", format="%Y-%m-%d")
        team_cond = (df_FIVB["Home_team"] == "DEN") | (df_FIVB["Away_team"] == "DEN")
        df_FIVB = df_FIVB[~(team_cond & date_cond)]

        ## NGR-TUN game 08.09.2021
        date_cond = df_FIVB["Date"] == pd.to_datetime("2021-09-08", format="%Y-%m-%d")
        team_cond = (df_FIVB["Home_team"] == "NGR") & (df_FIVB["Away_team"] == "TUN")
        df_FIVB.loc[date_cond & team_cond, "Venue"] = "RWA"

        ## BIH-ROU game 13.08.2022 (wrong dates on FIVB, should be 11.08.2022)
        date_cond = df_FIVB["Date"] == pd.to_datetime("2022-08-13", format="%Y-%m-%d")
        team_cond = (df_FIVB["Home_team"] == "BIH") & (df_FIVB["Away_team"] == "ROU")
        df_FIVB.loc[date_cond & team_cond, "Venue"] = "ROU"
        df_FIVB.loc[date_cond & team_cond, "Date"] = pd.to_datetime("2022-08-11", format="%Y-%m-%d")

        ## BIH-ALB game 17.08.2022 (wrong dates on FIVB, should be 06.08.2022)
        date_cond = df_FIVB["Date"] == pd.to_datetime("2022-08-17", format="%Y-%m-%d")
        team_cond = (df_FIVB["Home_team"] == "BIH") & (df_FIVB["Away_team"] == "ALB")
        df_FIVB.loc[date_cond & team_cond, "Venue"] = "BIH"
        df_FIVB.loc[date_cond & team_cond, "Date"] = pd.to_datetime("2022-08-06", format="%Y-%m-%d")

        ## eliminate games Mongolia forfeited in august 2023
        date_cond = ((df_FIVB["Date"] > pd.to_datetime("2023-08-18", format="%Y-%m-%d"))
                     & (df_FIVB["Date"] < pd.to_datetime("2023-08-25", format="%Y-%m-%d")))
        team_cond = (df_FIVB["Home_team"] == "MGL") | (df_FIVB["Away_team"] == "MGL")
        df_FIVB = df_FIVB[~(team_cond & date_cond)]

        ## eliminate games Uzbekistan and Pakistan forfeited in july 2023
        date_cond = ((df_FIVB["Date"] >= pd.to_datetime("2023-07-07", format="%Y-%m-%d"))
                     & (df_FIVB["Date"] <= pd.to_datetime("2023-07-12", format="%Y-%m-%d")))
        team_cond = (df_FIVB["Home_team"].isin({"UZB", "PAK"})) | (df_FIVB["Away_team"].isin({"UZB", "PAK"}))
        df_FIVB = df_FIVB[~(team_cond & date_cond)]

        # calculate the match weight
        res_to_score = {"Home_result": [3, 3, 3, 2, 1, 0], "Away_result": [0, 1, 2, 3, 3, 3], "Score": r_FIVB}
        Score = np.zeros(len(df_FIVB), float)
        for i in range(6):
            Score += ((df_FIVB["Home_result"] == res_to_score["Home_result"][i])
                      & (df_FIVB["Away_result"] == res_to_score["Away_result"][i])) * res_to_score["Score"][i]
        z = df_FIVB["Home_ranking"] - df_FIVB["Away_ranking"]

        scale = 125.0
        Qy = FIVB_calculation.Qy_CL(z.values / scale, c_FIVB)
        Expected_score = Qy @ r_FIVB
        g_FIVB = Expected_score - Score.values
        step = 0.01
        xi_calculated = np.abs(df_FIVB["Increment"].values / (g_FIVB * step * scale)).round(4)

        df_FIVB["Weight"] = xi_calculated
        df_FIVB["Category"] = -1
        xi_FIVB = [1.0, 1.75, 2.0, 3.5, 4.0, 4.5, 5.0]
        for v, xi in enumerate(xi_FIVB):
            fi = np.abs(xi_calculated-xi) < 0.12
            df_FIVB.loc[fi, "Weight"] = xi
            df_FIVB.loc[fi, "Category"] = v
        ## set manually the remaining results (all have category=3 and thus xi=xi_FIVB[3]=3.5)
        fi = (df_FIVB["Category"] == -1)
        df_FIVB.loc[fi, "Category"] = 3
        df_FIVB.loc[fi, "Weight"] = xi_FIVB[3]

        ## testing: recalculate the increment using the weights we guessed
        z_r = df_FIVB["Home_ranking"] - df_FIVB["Away_ranking"]
        g_FIVB_r = FIVB_calculation.Qy_CL(z_r.values / scale, c_FIVB) @ r_FIVB - Score.values
        df_FIVB["new_increment"] = g_FIVB_r * step * scale * df_FIVB["Weight"]

        ## manualy set the venues
        ## OLYMPIC GAMES 2021
        df_FIVB.loc[df_FIVB["Category"] == 6, "Venue"] = c2a["Japan"]

        df = df_FIVB
        ## Swap home <-> away for "away-venues"
        fi = (df["Away_team"] == df["Venue"])
        df_tmp = df.copy()
        df.loc[fi, "Home_team"] = df_tmp.loc[fi, "Away_team"]
        df.loc[fi, "Home_ranking"] = df_tmp.loc[fi, "Away_ranking"]
        df.loc[fi, "Home_result"] = df_tmp.loc[fi, "Away_result"]
        df.loc[fi, "Away_team"] = df_tmp.loc[fi, "Home_team"]
        df.loc[fi, "Away_ranking"] = df_tmp.loc[fi, "Home_ranking"]
        df.loc[fi, "Away_result"] = df_tmp.loc[fi, "Home_result"]

        # set HFA
        df["HFA"] = (df["Home_team"] == df["Venue"]) * 1

        # set the ordinal result
        result = np.zeros(len(df_FIVB), int)
        for i in range(6):
            result += ((df_FIVB["Home_result"] == res_to_score["Home_result"][i]) &
                       (df_FIVB["Away_result"] == res_to_score["Away_result"][i])) * i
        df.insert(7, "Result", result)

    # organize data for regression and return
    countries = list(set(df["Home_team"]).union(df["Away_team"]))
    countries.sort()
    M = len(countries)
    T = len(df)
    name2ind = pd.Series(np.arange(M), index=countries)
    ii = name2ind[df["Home_team"].values].values
    jj = name2ind[df["Away_team"].values].values

    XX = np.zeros((M, T))
    XX[ii, np.arange(T)] = 1
    XX[jj, np.arange(T)] = -1
    yy = df["Result"].values.astype(
        int)  ## to use full int representation (necessary in vmap when using yy for addressing)
    ## exogeneous variables here:
    uu = {}
    uu["countries"] = countries     # to help interpret the data in XX
    uu["name2ind"] = name2ind
    ## categories of the games
    uu["category"] = df["Category"].values
    ## HFA indicators
    uu["hfa"] = df["HFA"].values
    Data = (XX, yy, uu)
    if split:
        T_train = (df["Date"] < split_date).sum()
        XX_train = XX[:, :T_train]
        XX_val = XX[:, T_train:]
        yy_train = yy[:T_train]
        yy_val = yy[T_train:]
        uu_train = {}
        uu_val = {}
        for key in uu:
            uu_train[key] = uu[key][:T_train]
            uu_val[key] = uu[key][T_train:]
        return (XX_train, yy_train, uu_train), (XX_val, yy_val, uu_val)

    if return_theta:
        # finds theta given by FIVB for each point of time
        theta_0 = np.zeros(M)
        theta = np.zeros((M,T))
        t_start = np.zeros(M, int)
        t_end = np.zeros(M, int)
        for m in range(M):
            tt = np.argwhere(XX[m, :] != 0).ravel()
            t_start[m] = tt[0]
            t_end[m] = tt[-1]
            t_s = 0
            for t in tt:
                if XX[m, t] == -1:
                    skill = df["Away_ranking"].values[t]
                    at_Home = False
                else:
                    skill = df["Home_ranking"].values[t]
                    at_Home = True
                t_e = t+1
                theta[m, t_s: t_e] = skill
                t_s = t_e
            if t_e < T:
                won_Home = df["Result"].values[t] < 3           ## counts as home victory
                sign_increment = 1 - 2.0 * (won_Home ^ at_Home)  ## = -1 if lost @ home or won @ away
                skill += df["Increment"].values[t] * sign_increment
                theta[m, t_s:] = skill

        return df, Data, theta
    else:
        return df, Data

def find_theta_init(df):
    countries = set(df["Home_team"]).union(df["Away_team"])
    theta_0 = pd.Series(index=countries)
    for cc in countries:
        fi = len(df)
        fi_Home = df.index[df["Home_team"] == cc]
        if len(fi_Home) > 0:
            fi = fi_Home[0]
            theta = df.loc[fi, "Home_ranking"]
        fi_Away = df.index[df["Away_team"] == cc]
        if len(fi_Away) > 0 and (fi_Away[0] < fi):
            theta = df.loc[fi_Away[0], "Away_ranking"]

        theta_0[cc] = theta
    return theta_0

def empirical_entropy(k):
    # k : vector: number of games for each result
    T = k.sum()
    return -np.dot(k/T, np.log((k-1)/(T-1)))

def c2rtilde(cc):
    ## Calculate r_tilde: the new valus of numerical-score
    ## by making the first derivative of the implicit FIVB loss function the sams as log-score

    if isinstance(cc, list):
        return [c2rtilde(c) for c in cc]
    ## assumes cc is a shapless vector
    Lc = len(cc)
    pdf_cc = np.zeros(Lc + 2)
    pdf_cc[1:-1] = stats.norm.pdf(cc)
    cdf_cc = np.zeros(Lc + 2)
    cdf_cc[1:-1] = stats.norm.cdf(cc)
    cdf_cc[-1] = 1
    r_tilde = np.diff(pdf_cc) / np.diff(cdf_cc)
    r_tilde /= r_tilde[0]
    r_tilde *= 2.0
    return r_tilde

########################################################################
def AC_model(z, alpha, delta):
    v = alpha[None, :] + z[:, None] * delta[None, :]
    # negated log-loss
    ell = scipy.special.logsumexp(v, axis=1) - v
    val = np.exp(-ell)
    return val

########################################################################
c_FIVB_inf = np.array([-np.inf, -1.06, -0.394, 0, 0.394, 1.06, np.inf])
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
def main():

    ## to get results for the table with frequencies of the games, the FIVB results

    # df, data = import_FIVB_data()
    # theta_0 = find_theta_init(df)
    c_FIVB = np.array([-1.06, -0.394, 0, 0.394, 1.06])
    r_FIVB = np.array([2.0, 1.5, 1.0, -1.0, -1.5, -2.0])

    fig_dir = "/Users/leszek/Dropbox/Apps/Overleaf/FIVB.rating/figures/"

    def ell_fun(zz, params, der=0):
        if der == 0:
            ell = np.array([vmap(RegularizedRegression.logarithmic_loss_CL, in_axes=[0, None, None])
                            (zz, y, params) for y in range(6)]).T
        elif der == 1:
            ell = np.array([vmap(grad(RegularizedRegression.logarithmic_loss_CL), in_axes=[0, None, None])
                            (zz, y, params) for y in range(6)]).T
        return ell

    def ell_FIVB_fun(zz, rr):
        params.update({"r": rr, "Ar": np.eye(6)})
        ell = np.array([vmap(RegularizedRegression.fivb_loss, in_axes=[0, None, None])
                        (zz, y, params) for y in range(6)]).T
        return ell

    # print general statistics
    if False:
        df, data, theta_FIVB = import_FIVB_data(return_theta=True)
        X, y, u = data
        M, T = X.shape
        print("There are {} team, and {} games".format(M, T))
        k_hfa = df[df["HFA"] == 1]['Result'].value_counts().sort_index()
        k_ntr = df[df["HFA"] == 0]['Result'].value_counts().sort_index()
        k_ntr = 0.5 * (k_ntr.values + k_ntr.values[::-1])  # makes the neutral games symmetric
        T_ntr = k_ntr.sum()
        T_hfa = k_hfa.sum()
        k_hfa = k_hfa.values
        k_hfa_tilde = 0.5 * (k_hfa + k_hfa[::-1])
        # k_hfa *= 0
        print("k_ntr:", k_ntr)
        print("k_hfa:", k_hfa)
        print("k_hfa_tilde:", k_hfa_tilde)
        c_FIVB = np.array([-1.06, -0.394, 0.0, 0.394, 1.06])
        c_hat_all = stats.norm.ppf((k_hfa_tilde + k_ntr).cumsum() / (k_hfa_tilde + k_ntr).sum())
        print("c_y from all games :", c_hat_all)
        c_hat_ntr = stats.norm.ppf((k_ntr).cumsum() / T_ntr)
        print("c_y from neutral games :", c_hat_ntr)

        eta = np.linspace(0, 0.5, 100)
        Qy = FIVB_calculation.Qy_CL(eta, c_hat_ntr[:-1])
        val_FIVB = -np.log(Qy)
        LL = val_FIVB @ k_hfa
        print("eta from hfa games:", eta[LL.argmin()])
        if True:
            fig, ax = plt.subplots()
            ax.plot(eta, LL)
            ax.grid()
            ax.set_xlabel(r"$\eta$")
            ax.set_ylabel(r"$L(\eta)$", labelpad=0)
            # ax.set_ylim([0, 0.4])
            ax.set_xlim([0, eta[-1]])
            fig_name = "L(eta)_vs_eta.png"
            # plt.savefig(fig_dir + fig_name,  dpi=dpi_my, format="png")


        print("========\n log score when estimating from frequency: ")
        H_ntr = empirical_entropy(k_ntr)
        H_hfa = empirical_entropy(k_hfa)
        H_all = (H_ntr * T_ntr + H_hfa * T_hfa) / (T_ntr + T_hfa)
        print("neutral={:.2f}, home={:.2f}, all={:.2f}".format(H_ntr, H_hfa, H_all))
        print("corresponding probabilities: ")
        print("neutral={:.2f}, home={:.2f}, all={:.2f}".format(np.exp(-H_ntr), np.exp(-H_hfa), np.exp(-H_all)))

        print("=======")
        print("log-score for the FIVB algorithm")
        z = (df["Home_ranking"] - df["Away_ranking"]).values
        y = df["Result"].values
        scale = 125.0
        Qy = FIVB_calculation.Qy_CL(z / scale, c_FIVB)
        val_FIVB = -np.log(Qy[np.arange(T), y])
        val_FIVB_all = val_FIVB.mean()
        val_FIVB_ntr = val_FIVB[df["HFA"].values == 0].mean()
        val_FIVB_hfa = val_FIVB[df["HFA"].values == 1].mean()

        print("neutral={:.2f}, home={:.2f}, all={:.2f}".format(val_FIVB_ntr, val_FIVB_hfa, val_FIVB_all))
        print("corresponding proba for the FIVB algorithm")
        print("neutral={:.2f}, home={:.2f}, all={:.2f}".format(np.exp(-val_FIVB_ntr), np.exp(-val_FIVB_hfa),
                                                               np.exp(-val_FIVB_all)))
        print("=======")
        print("log-score for the FIVB algorithm + eta=0.2")
        hfa = df["HFA"].values * 0.2
        Qy = FIVB_calculation.Qy_CL(z / scale + hfa, c_FIVB)
        val_FIVB = -np.log(Qy[np.arange(T), y])
        val_FIVB_all = val_FIVB.mean()
        val_FIVB_ntr = val_FIVB[df["HFA"].values == 0].mean()
        val_FIVB_hfa = val_FIVB[df["HFA"].values == 1].mean()

        print("neutral={:.2f}, home={:.2f}, all={:.2f}".format(val_FIVB_ntr, val_FIVB_hfa, val_FIVB_all))
        print("corresponding proba for the FIVB algorithm")
        print("neutral={:.2f}, home={:.2f}, all={:.2f}".format(np.exp(-val_FIVB_ntr), np.exp(-val_FIVB_hfa),
                                                               np.exp(-val_FIVB_all)))

        print("======")  ## participation per period
        print("participation per period")
        cut_months = [6, 12]
        m_ii = []
        g_ii = []
        date_start = pd.to_datetime('2001-1-01', format="%Y-%m-%d")
        for year in range(2021, 2023 + 1):
            for month in cut_months:
                date_end = pd.to_datetime(f'{year}-{month}-01', format="%Y-%m-%d")
                fi = (df["Date"].values >= date_start) & (df["Date"].values < date_end)
                date_start = date_end
                g_ii.append(fi.sum())  # number of games per period
                m_ii.append((np.abs(X[:, fi]).sum(axis=1) > 0).sum())  # number of active teams
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
            tt = np.argwhere(X[m, :] != 0).ravel()
            first_game[m] = tt[0]
            last_game[m] = tt[-1]
            number_of_games[m] = len(tt)
        countries = u["countries"]
        mi = first_game.argmax()
        print(f"the latest first game of {countries[mi]} on :", df.loc[first_game[mi], "Date"])
        mi = last_game.argmin()
        print(f"the earliest last game of {countries[mi]} on :", df.loc[last_game[mi], "Date"])
        ng = number_of_games.min()
        print(f"smallest number of games by {np.array(countries)[ng == number_of_games]}:", ng)
        ng = number_of_games.max()
        print(f"largest number of games by {np.array(countries)[ng == number_of_games]}:", ng)

    # figure showing Q_y(z) and ell_y(z)
    if False:
        dpi_my = 600  ## resolution
        HIDE_ell_tilde = False   ## do not show the adapted results
        c_CL = c_FIVB
        ## find good r_tilde (for z_0 =0)
        r_tilde = c2rtilde(c_CL)

        zz = np.linspace(-3,3,1000)

        params = {"CDF": 0, "c": c_CL, "Ac": np.eye(5)}
        ell = ell_fun(zz, params)
        ell_0 = ell_fun(np.zeros(1), params)
        Qy = np.exp(-ell)
        dot_ell_0 = ell_fun(np.zeros(1), params, der=1)
        dot_ell_0 = dot_ell_0.flatten()


        sigma = np.sqrt(np.pi/8)
        params = {"CDF": 1, "c": c_CL/sigma, "Ac": np.eye(5)}
        ell_logistic = ell_fun(zz/sigma, params)
        Qy_logistic = np.exp(-ell_logistic)

        LABELS=["``3-0''", "``3-1''", "``3-2''", "``2-3''", "``1-3''", "``0-3''"]
        COLOR = ["g", "r", "m", "b", "k", "c"]
        MARKERS=["o", "P", "<", "D", "v", "*"]

        ###### plot ell(z)
        r_y = np.array([2.0, 1.5, 1.0, -1.0, -1.5, -2.0])
        # rhat = Qy @ r_y[:, None]

        ell_FIVB = ell_FIVB_fun(zz, r_y)
        ell_FIVB_0 = ell_FIVB_fun(np.zeros(1), r_y)
        ## vertical shift and scaling to keep the same values for z=0
        alpha = -dot_ell_0[0]/r_y[0]
        b_y = ell_0 - alpha * ell_FIVB_0
        ell_FIVB = alpha * ell_FIVB + b_y

        ell_FIVB_tilde = ell_FIVB_fun(zz, r_tilde)
        ell_FIVB_tilde_0 = ell_FIVB_fun(np.zeros(1), r_tilde)
        ## vertical shift and scaling to keep the same values for z=0
        alpha_tilde = -dot_ell_0[0] / r_tilde[0]
        b_y_tilde = ell_0 - alpha * ell_FIVB_tilde_0
        ell_FIVB_tilde = alpha_tilde * ell_FIVB_tilde + b_y_tilde

        MS = 10
        (fig, ax) = plt.subplots()
        for k in range(ell.shape[1]//2):
            ax.plot(zz, ell[:, k], ls="-", color=COLOR[k], label=LABELS[k], marker=MARKERS[k], ms=MS, markevery=100)
            ax.plot(zz, ell_FIVB[:, k], ls="--", color=COLOR[k], marker=MARKERS[k], ms=MS, markevery=100, mfc="w")
            if not HIDE_ell_tilde:
                ax.plot(zz, ell_FIVB_tilde[:, k], ls=(0,(5,3,1,3)), color=COLOR[k], ms=MS, marker=MARKERS[k], markevery=100,
                    mfc="w")
            if False: #k > 0:  ## plot big marker at the minimum of the function
                kk_min = np.argmin(ell[:,k])
                ax.plot(zz[kk_min], ell[kk_min,k], color=COLOR[k], marker=MARKERS[k], markersize=12)
                kk_min = np.argmin(ell_FIVB[:, k])
                ax.plot(zz[kk_min], ell_FIVB[kk_min, k], color=COLOR[k], marker=MARKERS[k], markerfacecolor="w", markersize=12)
        ax.set_xlabel("$z$")
        ax.set_ylabel(r"$\ell_y(z)$")
        ax.grid()
        ax.set_ylim([0, 4.1])
        ax.set_xlim([-1, 3])
        legend1 = ax.legend(loc='upper right')

        legend_elements2 = [
            Line2D([0], [0], color='k', label=r'$\ell_y(z)$', ls="-"),
            Line2D([0], [0], color='k', label=r'$a\ell^{\textnormal{FIVB}}_y(z;\boldsymbol{r}^{\textnormal{FIVB}})+b_y$', ls="--"),
        ]
        if not HIDE_ell_tilde:
            legend_elements2.append(Line2D([0], [0], color='k', label=r'$\tilde{a}\ell^{\textnormal{FIVB}}_y(z; \tilde{\boldsymbol{r}})+\tilde{b}_y$', ls=(0,(5,3,1,3))))
        ax.legend(handles=legend_elements2, loc='upper left')
        ax.add_artist(legend1)
        fig.tight_layout()

        fig_name = "ell_y_z.png"
        if HIDE_ell_tilde:
            fig_name = "ell_y_z_no_tilde.png"

        # plt.savefig(fig_dir + fig_name, dpi=dpi_my, format="png")

        ############## plot Qy(z) for probit and logit
        POS = [(2, 0.8), (1.8, 0.2), (0.8, 0.15), (-1, 0.15), (-2, 0.2), (-1.9, 0.8)]
        (fig, ax) = plt.subplots()
        for k, lab in enumerate(LABELS):
            ax.plot(zz, Qy[:, k], color=COLOR[k])
            ax.text(POS[k][0], POS[k][1], LABELS[k])
        ax.set_xlabel("$z$")
        ax.set_ylabel(r"$\mathsf{P}^{\textnormal{FIVB}}_y(z)$")
        ax.grid()
        ax.set_ylim([0, 1.1])
        ax.set_xlim([-3, 3])

        fig_name = "P_y_z.png"
        # plt.savefig(fig_dir + fig_name,  dpi=dpi_my, format="png")

        ############## plot Qy(z) for probit and logit
        (fig, ax) = plt.subplots()
        POS = [(0.6, 0.3), (1.6, 0.2), (0.8, 0.15), (-1, 0.15), (-1.8, 0.2), (-1., 0.3)]
        for k, lab in enumerate(LABELS):
            ax.plot(zz, Qy[:, k], color=COLOR[k])
            ax.plot(zz, Qy_logistic[:, k], color=COLOR[k], ls="--")
            ax.text(POS[k][0], POS[k][1], LABELS[k])
        ax.set_xlabel("$z$")
        ax.set_ylabel(r"$\mathsf{P}_y(z), \mathsf{P}^{\mathcal{L}}_y(z/\sigma)$")
        ax.grid()
        ax.set_ylim([0, 0.4])
        ax.set_xlim([-3, 3])

        ## custom legend
        legend_elements = [
            Line2D([0], [0], ls="-", color='k', label=r'$\mathsf{P}_y\big(z; \boldsymbol{c}^{\text{FIVB}}\big)$'),
            Line2D([0], [0], ls="--", color='k', label=r'$\mathsf{P}^{\mathcal{L}}_y\big(z/\sigma; \boldsymbol{c}^{\text{FIVB}}/\sigma$\big)')
        ]
        ax.legend(handles=legend_elements, loc='upper left')

        fig_name = "P_y_z_Gauss.vs.logistic.png"
        # plt.savefig(fig_dir + fig_name,  dpi=dpi_my, format="png")

    # Figures showing ALO + figure showing optimized c and eta and c^FIVB
    if False:
        ## load data
        save_DIR = "results/FIVB/"
        # train_ext = "_train"
        train_ext = ""
        ext = train_ext + ".npz"

        ticks_transform_function = lambda x: np.exp(-x) * 100

        save_file = "alo_logloss_eta0"
        res = np.load(save_DIR + save_file + ext, allow_pickle=True)
        alo_logloss_eta0 = res["alo"].reshape((1,))[0]
        params_FIVB = res["params"]

        save_file = "alo_logloss_opt_c"
        res = np.load(save_DIR + save_file + ext, allow_pickle=True)
        alo_logloss_opt_c = res["alo"].reshape((1,))[0]
        pp_opt_c = res["params"]

        save_file = "alo_logloss_opt_eta"
        res = np.load(save_DIR + save_file + ext, allow_pickle=True)
        alo_logloss_opt_eta = res["alo"].reshape((1,))[0]
        pp_opt_eta = res["params"]

        save_file = "alo_logloss_opt_c_eta"
        res = np.load(save_DIR + save_file + ext, allow_pickle=True)
        alo_logloss_opt_c_eta = res["alo"].reshape((1,))[0]
        pp_opt_c_eta = res["params"]

        save_file = "alo_FIVB_eta0"
        res = np.load(save_DIR + save_file + ext, allow_pickle=True)
        alo_FIVB_eta0 = res["alo"].reshape((1,))[0]
        pp_FIVB_eta0 = res["params"]

        save_file = "alo_FIVB_opt_r_eta0"
        res = np.load(save_DIR + save_file + ext, allow_pickle=True)
        alo_FIVB_opt_r = res["alo"].reshape((1,))[0]
        pp_FIVB_opt_r = res["params"]

        save_file = "alo_FIVB_opt_r_eta"
        res = np.load(save_DIR + save_file + ext, allow_pickle=True)
        alo_FIVB_opt_r_eta = res["alo"].reshape((1,))[0]
        pp_FIVB_opt_r_eta = res["params"]

        save_file = "alo_FIVB_opt_c_r_eta"
        res = np.load(save_DIR + save_file + ext, allow_pickle=True)
        alo_FIVB_opt_c_r_eta = res["alo"].reshape((1,))[0]
        pp_FIVB_opt_c_r_eta = res["params"]

        save_file = "alo_FIVB_r_tilde_eta_02"
        res = np.load(save_DIR + save_file + ext, allow_pickle=True)
        alo_FIVB_r_tilde_eta_02 = res["alo"].reshape((1,))[0]
        pp_FIVB_r_tilde_eta_02 = res["params"]

        save_file = "alo_FIVB_opt_c_r_tilde_eta_02"
        res = np.load(save_DIR + save_file + ext, allow_pickle=True)
        alo_FIVB_opt_c_r_tilde_eta_02 = res["alo"].reshape((1,))[0]
        pp_FIVB_opt_c_r_tilde_eta_02 = res["params"]

        save_file = "alo_logloss_FIVB_weight_eta_02"
        res = np.load(save_DIR + save_file + ext, allow_pickle=True)
        alo_logloss_FIVB_weight_eta_02 = res["alo"].reshape((1,))[0]

        save_file = "alo_logloss_opt_weight_eta"
        res = np.load(save_DIR + save_file + ext, allow_pickle=True)
        alo_logloss_opt_weight_eta = res["alo"].reshape((1,))[0]
        pp_logloss_opt_weight_eta = res["params"]

        save_file = "alo_logloss_CL_logistic_tilde_opt_c_eta"   ## logistic cdf but parameters obtained from Gaussian optimized model
        res = np.load(save_DIR + save_file + ext, allow_pickle=True)
        alo_logloss_CL_logistic_tilde_opt_c_eta = res["alo"].reshape((1,))[0]
        pp_logloss_CL_logistic_tilde_opt_c_eta = res["params"]

        # save_file = "alo_logloss_CL_logistic_tilde_FIVB_c_eta0"
        # res = np.load(save_DIR + save_file + ext, allow_pickle=True)
        # alo_logloss_CL_logistic_tilde_FIVB_c_eta0 = res["alo"].reshape((1,))[0]

        save_file = "alo_logloss_CL_logistic_tilde_FIVB_c_opt_eta"
        res = np.load(save_DIR + save_file + ext, allow_pickle=True)
        alo_logloss_CL_logistic_tilde_FIVB_c_opt_eta = res["alo"].reshape((1,))[0]

        save_file = "alo_logloss_CL_logistic_opt_c_eta"
        res = np.load(save_DIR + save_file + ext, allow_pickle=True)
        alo_logloss_CL_logistic_opt_c_eta = res["alo"].reshape((1,))[0]
        pp_logloss_CL_logistic_opt_c_eta = res["params"]


        ## plot figures
        COLOR = ["r", "g", "m"]
        MARKERS = ["o", "v", "P", 's', '*', 'H']
        LABELS = ["all", "ntr", "hfa"]
        i2N = ["full", "neutral", "hfa"]
        MS = 10
        MFC = ["k", "w", "m"]
        MEC = ["k", "g", "w"]
        LS = ["-", "--", "-."]

        # ax.semilogx(gamma_vec, loo, color='k', zorder=1) # , marker="x")
        gamma_vec = np.array([pp["gamma"] for pp in params_FIVB])

        ###########################################################################
        ###########################################################################
        #### FIGURE to show optimization of c and eta
        fig, ax = plt.subplots()
        ax.set_xscale("log")
        # logloss references
        LIST_results = [1, 2]
        for i in LIST_results:
            ax.plot(gamma_vec, alo_logloss_eta0[i2N[i]], ls=LS[i], c=COLOR[i], mec=COLOR[i], marker=MARKERS[0],
                    mfc=MFC[i], ms=MS, zorder=2, markevery=(1,2))

        for i in LIST_results:
            ax.plot(gamma_vec, alo_logloss_opt_c[i2N[i]], ls=LS[i], c=COLOR[i], mec=COLOR[i], marker=MARKERS[1],
                    mfc=MFC[i], ms=MS, zorder=2, markevery=(1,2))

        for i in LIST_results:
            ax.plot(gamma_vec, alo_logloss_opt_eta[i2N[i]], ls=LS[i], c=COLOR[i], mec=COLOR[i], marker=MARKERS[3],
                    mfc=MFC[i], ms=MS, zorder=2, markevery=(0,2))

        for i in LIST_results:
            ax.plot(gamma_vec, alo_logloss_opt_c_eta[i2N[i]], ls=LS[i], c=COLOR[i], mec=COLOR[i], marker=MARKERS[2],
                    mfc=MFC[i], ms=MS, zorder=2, markevery=2)

        ax.grid()
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$U^{\text{ntr}}, U^{\text{hfa}}$", labelpad=0)
        y_limit = [1.32, 1.42]
        ax.set_ylim(y_limit)
        ax.set_xlim([gamma_vec[0], gamma_vec[-1]])
        ## plot second ax which interprets the log-likelihood as probability

        ax2 = ax.twinx()

        ax2.set_yticks(ax.get_yticks())
        ax2.set_yticklabels(ticks_transform_function(ax.get_yticks()).round(1).astype(str).tolist())
        ax2.set_ylabel(r"$V^{\text{ntr}}, V^{\text{hfa}}$ [\%]", labelpad=3)
        ax2.set_ylim(ax.get_ylim())

        ## custom legend
        legend_elements = [
            Line2D([0], [0], marker=MARKERS[0], color='k', label=r'$\boldsymbol{c}^{\text{FIVB}}, \eta=0$',
                   mfc='w', mec="k", ms=MS),
            Line2D([0], [0], marker=MARKERS[1], color='k', label=r'$\hat{\boldsymbol{c}}, \eta=0$',
                   mfc='w', mec="k", ms=MS),
            Line2D([0], [0], marker=MARKERS[3], color='k', label=r'$\boldsymbol{c}^{\text{FIVB}}, \hat\eta$',
                   mfc='w', mec="k", ms=MS),
            Line2D([0], [0], marker=MARKERS[2], color='k', label=r'$\hat{\boldsymbol{c}}, \hat{\eta}$',
                   mfc='w', mec="k",  ms=MS)
        ]
        ax.legend(handles=legend_elements, loc='upper left')

        fig.tight_layout()
        fig_name = "ALO_logloss.png"
        # plt.savefig(fig_dir + fig_name,  dpi=dpi_my, format="png")

        ###########################################################################
        ###########################################################################
        # Show the optimized thresholds "c" and HFA "eta"COLOR
        fig, ax = plt.subplots()
        ax.set_xscale("log")
        c_opt = np.array([pp["c"] for pp in pp_opt_c_eta])
        eta_opt = np.array([pp["eta"] for pp in pp_opt_c_eta])
        c_init = params_FIVB[0]["c"]

        i = 2
        LAB = r"$\hat\eta$"
        ax.plot(gamma_vec, eta_opt, linestyle=LS[i], color=COLOR[i], label=LAB)

        for i in range(2):
            LAB  = r"$\hat{c}_" + str(i) + "$"
            ax.plot(gamma_vec, c_opt[:, i], linestyle=LS[i], color=COLOR[i], label=LAB)
            # ax.fill_between(v_vec, C[:, i] + 2.0*STD[:, i], C[:, i] - 2.0*STD[:, i], linestyle="--", color=COLOR[i], alpha=0.2)
            # ax.plot(gamma_vec, C_tilde[:, i], linestyle="-.", color="k")
            # show the FIVB values
            LAB = r"$c^{\text{FIVB}}_" + str(i) + "$"
            ax.plot(gamma_vec, c_init[i] * np.ones_like(gamma_vec), ':', color=COLOR[i], label=LAB)

        ax.grid()
        ax.set_xlabel(r"$\gamma$")
        # ax.set_ylabel("$c_0, c_1, \\eta$", labelpad=0)
        ax.set_ylim([-1.1, 0.4])
        ax.set_xlim([gamma_vec[0], gamma_vec[-1]])
        ax.legend(loc='upper left')
        fig_name = "c_eta_vs_gamma.png"
        # plt.savefig(fig_dir + fig_name,  dpi=dpi_my, format="png")

        ###########################################################################
        ###########################################################################
        #### FIGURE to show FIVB-model and optimization of r
        fig, ax = plt.subplots()
        ax.set_xscale("log")

        LIST_results = [1, 2]

        for i in LIST_results:
            ax.plot(gamma_vec, alo_FIVB_eta0[i2N[i]], ls=LS[i], c=COLOR[i], mec=COLOR[i], marker=MARKERS[3],
                    mfc=MFC[i], ms=MS, zorder=2, markevery=2)

            ax.plot(gamma_vec, alo_FIVB_opt_r_eta[i2N[i]], ls=LS[i], c=COLOR[i], mec=COLOR[i], marker=MARKERS[0],
                    mfc=MFC[i], ms=MS, zorder=2, markevery=(0,2))

            ax.plot(gamma_vec, alo_FIVB_r_tilde_eta_02[i2N[i]], ls=LS[i], c=COLOR[i], mec=COLOR[i], marker=MARKERS[1],
                    mfc=MFC[i], ms=MS, zorder=2, markevery=(1,2))

            ax.plot(gamma_vec, alo_FIVB_opt_c_r_tilde_eta_02[i2N[i]], ls=LS[i], c=COLOR[i], mec=COLOR[i], marker=MARKERS[2],
                    mfc=MFC[i], ms=MS, zorder=2, markevery=(0,2))

            ax.plot(gamma_vec, alo_FIVB_opt_c_r_eta[i2N[i]], ls=LS[i], c=COLOR[i], mec=COLOR[i], marker=MARKERS[4],
                    mfc=MFC[i], ms=MS, zorder=2, markevery=(1,2))

            # ax.plot(gamma_vec, alo_logloss_opt_c_eta[i2N[i]], ls=LS[i], c=COLOR[i], mec=COLOR[i], marker=MARKERS[2],
            #         mfc=MFC[i], ms=MS, zorder=2)

        ## custom legend
        legend_elements = [
            Line2D([0], [0], marker=MARKERS[3], color='k',
                   label=r'FIVB, $\boldsymbol{c}^{\text{FIVB}}, \boldsymbol{r}^{\text{FIVB}}, \eta=0$',
                   mfc='w', mec="k", ms=MS),
            Line2D([0], [0], marker=MARKERS[0], color='k',
                   label=r'FIVB, $\boldsymbol{c}^{\text{FIVB}}, \hat{\boldsymbol{r}}, \hat{\eta}$',
                   mfc='w', mec="k", ms=MS),
            Line2D([0], [0], marker=MARKERS[1], color='k',
                   label=r"FIVB, $\boldsymbol{c}^{\text{FIVB}}, \tilde{\boldsymbol{r}}\big(\boldsymbol{c}^{\text{FIVB}}\big), \eta=0.2$",
                   mfc='w', mec="k", ms=MS),
            Line2D([0], [0], marker=MARKERS[4], color='k',
                   label=r'FIVB, $\hat{\boldsymbol{c}}, \hat{\boldsymbol{r}}, \hat{\eta}$',
                   mfc='w', mec="k", ms=MS),
            Line2D([0], [0], marker=MARKERS[2], color='k',
                   label=r'FIVB, $\hat{\boldsymbol{c}}, \tilde{\boldsymbol{r}}\big(\hat{\boldsymbol{c}}\big), \eta=0.2$',
                   mfc='w', mec="k", ms=MS)
        ]

        ax.grid()
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$U^{\text{ntr}}, U^{\text{hfa}}$", labelpad=0)
        ax.set_ylim(y_limit)
        ax.set_xlim([gamma_vec[0], gamma_vec[-1]])
        ## plot second ax which interprets the log-likelihood as probability

        ax2 = ax.twinx()
        ax2.set_yticks(ax.get_yticks())


        ax2.set_yticklabels(ticks_transform_function(ax.get_yticks()).round(1).astype(str).tolist())

        ax2.set_ylabel(r"$V^{\text{ntr}}, V^{\text{hfa}}$ [\%]", labelpad=3)
        ax2.set_ylim(ax.get_ylim())
        ax.legend(handles=legend_elements, loc='upper left')
        fig.tight_layout()
        fig_name = "ALO_FIVB.png"
        # plt.savefig(fig_dir + fig_name,  dpi=dpi_my, format="png")

        ###########################################################################
        ###########################################################################
        # show the numerical score optimized: r, and calculated from c: r_tilde
        fig, ax = plt.subplots()
        ax.set_xscale("log")
        r_FIVB = np.array([pp["r"] for pp in params_FIVB])
        cc = [pp["c"]  @ pp["Ac"].T for pp in params_FIVB]
        r_tilde = np.array(c2rtilde(cc=cc))
        r_opt_eta = np.array([pp["r"] for pp in pp_FIVB_opt_r_eta])


        ### plot FIVB r for reference
        ax.plot(gamma_vec, r_FIVB[:, 0], ls=LS[0], c="k")   # no marker
        for i in [1, 2]:
            ax.plot(gamma_vec, r_FIVB[:, i], ls=LS[i], c="k", marker="x", ms=MS)
        for i in [1, 2]:
            ax.plot(gamma_vec, r_tilde[:, i], ls=LS[i], c=COLOR[i], marker=MARKERS[0], mfc="w", ms=MS)
            ax.plot(gamma_vec, r_opt_eta[:, i], ls=LS[i], c=COLOR[i], marker=MARKERS[2], mfc="w", ms=MS)

        ax.grid()
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$r_l$", labelpad=0)
        ax.set_ylim([-0.5, 2.1])
        # ax.set_xlim([gamma_vec[0], gamma_vec[-1]])
        ## custom legend
        legend_elements = [
            Line2D([0], [0], marker="x", color='k', label=r'$r_l^{\text{FIVB}}$',
                   mfc='w', mec="k", ms=MS),
            Line2D([0], [0], marker=MARKERS[0], color='k', label=r'$\tilde{r}_l(\boldsymbol{c}^{\text{FIVB}})$',
                   mfc='w', mec="k", ms=MS),
            Line2D([0], [0], marker=MARKERS[2], color='k', label=r'$\hat{r}_l(\boldsymbol{c}^{\text{FIVB}})$',
                   mfc='w', mec="k", ms=MS)
        ]
        legend1 = ax.legend(handles=legend_elements, loc='upper left')
        legend_elements2 = [
            Line2D([0], [0], color='k', label=r'$r_0$', ls=LS[0]),
            Line2D([0], [0], color='k', label=r'$r_1$', ls=LS[1]),
            Line2D([0], [0], color='k', label=r'$r_2$', ls=LS[2])
        ]
        ax.legend(handles=legend_elements2, loc='upper right')
        ax.add_artist(legend1)
        fig.tight_layout()
        fig_name = "r_vs_gamma.png"
        # plt.savefig(fig_dir + fig_name,  dpi=dpi_my, format="png")

        # I verified here if the expected score was mononotonically increasing with $z$ for optimized \hat\br
        # zz = np.linspace(-3, 3, 1000)
        # Qy = np.exp(-ell_fun(zz, pp_FIVB_opt_r_eta[0]))
        # r_y = np.array([2.0, 1.5, 1.0, -1.0, -1.5, -2.0])
        # rhat = Qy @ r_y[:, None]

        ##########################################################################
        ###########################################################################
        #### FIGURE to show the effect of weights
        fig, ax = plt.subplots()
        ax.set_xscale("log")
        # logloss references
        LIST_results = [1, 2]
        for i in LIST_results:
            ax.plot(gamma_vec, alo_logloss_opt_eta[i2N[i]], ls=LS[i], c=COLOR[i], mec=COLOR[i], marker=MARKERS[3],
                    mfc=MFC[i], ms=MS, zorder=2, markevery=(0,2))
        for i in LIST_results:
            ax.plot(gamma_vec, alo_logloss_FIVB_weight_eta_02[i2N[i]], ls=LS[i], c=COLOR[i], mec=COLOR[i], marker=MARKERS[2],
                    mfc=MFC[i], ms=MS, zorder=2, markevery=(0,2))
        for i in LIST_results:
            ax.plot(gamma_vec, alo_logloss_opt_weight_eta[i2N[i]], ls=LS[i], c=COLOR[i], mec=COLOR[i], marker=MARKERS[4],
                    mfc=MFC[i], ms=MS, zorder=2, markevery=(1,2))

        ax.grid()
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$U^{\text{ntr}}, U^{\text{hfa}}$", labelpad=0)
        y_limit = [1.32, 1.42]
        ax.set_ylim(y_limit)
        ax.set_xlim([gamma_vec[0], gamma_vec[-1]])
        ## plot second ax which interprets the log-likelihood as probability
        ax2 = ax.twinx()
        ax2.set_yticks(ax.get_yticks())
        ax2.set_yticklabels(ticks_transform_function(ax.get_yticks()).round(1).astype(str).tolist())
        ax2.set_ylabel(r"$V^{\text{ntr}}, V^{\text{hfa}}$ [\%]", labelpad=3)
        ax2.set_ylim(ax.get_ylim())

        ## custom legend
        legend_elements = [
            Line2D([0], [0], marker=MARKERS[3], color='k', label=r'$\xi_v\equiv1, \hat\eta$',
                   mfc='w', mec="k", ms=MS),
            Line2D([0], [0], marker=MARKERS[2], color='k', label=r'$\xi^{\text{FIVB}}_v, \eta=0.2$',
                   mfc='w', mec="k", ms=MS),
            Line2D([0], [0], marker=MARKERS[4], color='k', label=r'$\hat{\xi}_v, \hat{\eta}$',
                   mfc='w', mec="k", ms=MS)
        ]
        ax.legend(handles=legend_elements, loc='upper left')

        fig.tight_layout()
        fig_name = "ALO_logloss_weights.png"
        # plt.savefig(fig_dir + fig_name,  dpi=dpi_my, format="png")

        ###########################################################################
        ###########################################################################
        ##### Figure shows the weights optimized
        fig, ax = plt.subplots()
        ax.set_xscale("log")
        ax.set_yscale("log")
        xi_opt = np.array([pp["weight"] for pp in pp_logloss_opt_weight_eta])
        eta_opt = np.array([pp["eta"] for pp in pp_logloss_opt_weight_eta])

        legend_elements = []
        for i in range(7):
            ax.plot(gamma_vec, xi_opt[:, i], ls=LS[0])
            legend_elements.append(Line2D([0], [0], color='k', label=r"$\xi_{"+str(i)+"}$"))

        ax.grid()
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$\xi_v$", labelpad=0)
        ax.set_ylim([0.8, 10])
        ax.set_xlim([gamma_vec[0], gamma_vec[-1]])

        legend1 = ax.legend(handles=legend_elements, loc='upper left')
        fig.tight_layout()
        fig_name = "xi_vs_gamma.png"
        # plt.savefig(fig_dir + fig_name,  dpi=dpi_my, format="png")

        ################################################################
        ################################################################
        ### show the ALO results for logistic_CL
        fig, ax = plt.subplots()
        ax.set_xscale("log")

        LIST_results = [1, 2]

        for i in LIST_results:
            ax.plot(gamma_vec, alo_logloss_opt_eta[i2N[i]], ls=LS[i], c=COLOR[i], mec=COLOR[i], marker=MARKERS[3],
                    mfc=MFC[i], ms=MS, zorder=2, markevery=(1,2))

        for i in LIST_results:
            ax.plot(gamma_vec, alo_logloss_CL_logistic_tilde_FIVB_c_opt_eta[i2N[i]], ls=LS[i], c=COLOR[i], mec=COLOR[i], marker=MARKERS[0],
                    mfc=MFC[i], ms=MS, zorder=2, markevery=(2,3))

        for i in LIST_results:
            ax.plot(gamma_vec, alo_logloss_opt_c_eta[i2N[i]], ls=LS[i], c=COLOR[i], mec=COLOR[i], marker=MARKERS[2],
                    mfc=MFC[i], ms=MS, zorder=2, markevery=(1,3))

        for i in LIST_results:
            ax.plot(gamma_vec, alo_logloss_CL_logistic_tilde_opt_c_eta[i2N[i]], ls=LS[i], c=COLOR[i], mec=COLOR[i], marker=MARKERS[1],
                    mfc=MFC[i], ms=MS, zorder=2, markevery=(0,3))

        for i in LIST_results:
            ax.plot(gamma_vec, alo_logloss_CL_logistic_opt_c_eta[i2N[i]], ls=LS[i], c=COLOR[i], mec=COLOR[i],
                    marker=MARKERS[4],
                    mfc=MFC[i], ms=MS, zorder=2, markevery=(2, 3))

        ax.grid()
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$U^{\text{ntr}}, U^{\text{hfa}}$", labelpad=0)
        ax.set_ylim(y_limit)
        ax.set_xlim([gamma_vec[0], gamma_vec[-1]])
        ax2 = ax.twinx()

        ax2.set_yticks(ax.get_yticks())
        ax2.set_yticklabels(ticks_transform_function(ax.get_yticks()).round(1).astype(str).tolist())
        ax2.set_ylabel(r"$V^{\text{ntr}}, V^{\text{hfa}}$ [\%]", labelpad=3)
        ax2.set_ylim(ax.get_ylim())
        ## custom legend
        legend_elements = [
            Line2D([0], [0], marker=MARKERS[3], color='k', label=r'$\Phi, \boldsymbol{c}^{\text{FIVB}}, \hat{\eta}$',
                   mfc='w', mec="k", ms=MS),
            Line2D([0], [0], marker=MARKERS[0], color='k', label=r'$\mathcal{L}, \boldsymbol{c}^{\text{FIVB}}/\sigma, \hat{\eta}/\sigma$',
                   mfc='w', mec="k", ms=MS),
            Line2D([0], [0], marker=MARKERS[2], color='k', label=r'$\Phi, \hat{\boldsymbol{c}}, \hat{\eta}$',
                   mfc='w', mec="k", ms=MS),
            Line2D([0], [0], marker=MARKERS[1], color='k', label=r'$\mathcal{L}, \hat{\boldsymbol{c}}/\sigma, \hat{\eta}/\sigma$',
                   mfc='w', mec="k", ms=MS),
            Line2D([0], [0], marker=MARKERS[4], color='k', label=r'$\mathcal{L}, \hat{\boldsymbol{c}}^{\mathcal{L}}, \hat{\eta}^{\mathcal{L}}$',
                   mfc='w', mec="k", ms=MS)
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        fig.tight_layout()
        fig_name = "ALO_CL.logit.vs.probit.png"
        # plt.savefig(fig_dir + fig_name,  dpi=dpi_my, format="png")

        None

    # Table/Figure showing real-time adaptation
    if True:
        def load_data_and_calculate(save_file):
            print_all = False
            print_table = True
            print_table_skills = True
            plot_skills = False
            print("results for ",save_file)
            res = np.load(save_DIR + save_file + ext, allow_pickle=True)
            pred_list = res["pred"]
            steps = res["steps"]
            U = {}
            for ii in ii_list:
                U["all"] = np.zeros(len(pred_list))
                U["hfa"] = np.zeros(len(pred_list))
                U["ntr"] = np.zeros(len(pred_list))
                ii_hfa = ii[hfa[ii] == 1]
                ii_ntr = ii[hfa[ii] == 0]
                for k, pred in enumerate(pred_list):
                    U["all"][k] = pred[ii].mean()
                    U["hfa"][k] = pred[ii_hfa].mean()
                    U["ntr"][k] = pred[ii_ntr].mean()
                if print_all:
                    print(f"for ii: {ii[0]}-{ii[-1]}")
                    print("all:", U["all"])
                    print("ntr:", U["ntr"])
                    print("hfa:", U["hfa"])

            i_opt = np.argmin(U["all"])
            th = res["theta"]
            if plot_skills:
                colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
                for ic, m in enumerate(m_list):
                    plt.plot_date(df["Date"], th[0][m, :], color=colors[ic], ls="-")
                plt.legend(c_list)
                for ic, m in enumerate(m_list):
                    plt.plot_date(df["Date"], th[i_opt][m, :], color=colors[ic], ls="--")
                plt.xticks(rotation=70)
            rho0 = np.array([stats.spearmanr(theta_FIVB[:, t+1], th[0][:, t])[0] for t in range(T-1)])
            rho = np.array([stats.spearmanr(theta_FIVB[:, t+1], th[i_opt][:, t])[0] for t in range(T-1)])
            if print_all:
                print("average Spearman correlation between FIVB data and nominal step size:", rho0.mean())
                print("average Spearman correlation between FIVB data and optimal step size:", rho.mean())

            if print_table:
                print("& {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}\\\\".format(steps[0],
                                                                                 U["all"][0], U["ntr"][0],U["hfa"][0],
                                                                                 rho0.mean()))
                print("& {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}\\\\".format(steps[i_opt],
                                                                             U["all"][i_opt], U["ntr"][i_opt], U["hfa"][i_opt],
                                                                             rho.mean()))
            if print_table_skills:
                for ii in [0, i_opt]:
                    print("step=",steps[ii])
                    th_opt = th[ii]
                    th_last = th_opt[:,-1]
                    indices = np.argsort(th_last)[::-1]
                    Imax = 7
                    names_sorted = np.array(countries)[indices[:Imax]]
                    th_sorted = th_last[indices[:Imax]]
                    for i in range(Imax):
                        print("& {:s} ".format(names_sorted[i]), end="")
                    print("\\\\")
                    for i in range(Imax):
                        print("& {:.1f} ".format(th_sorted[i]), end="")
                    print("\\\\")

            return res

        print("=============")
        print("evaluation of the RT estimation")
        ## load data
        df, data, theta_FIVB = import_FIVB_data(split=False, return_theta=True)
        X, y, u = data
        M, T = X.shape

        save_DIR = "results/FIVB/"
        ext = ".npz"
        split_date_str = ["2023-01-01", "2023-07-01"]
        split_date_list = [pd.to_datetime(str, format="%Y-%m-%d") for str in split_date_str]
        split_date_list.append(df["Date"].values[-1])  ## add the last one
        ii_list = []
        # ii_list = [ np.arange((df["Date"]<=split_date_list[k]).sum(),(df["Date"]<=split_date_list[k+1]).sum()) for k in range(len(split_date_list)-1)]
        ii_list.append(np.arange(T))
        hfa = u["hfa"]

        c_list = ["POL", "BRA", "BEL", "THA", "GER"]
        m_list = u["name2ind"][c_list].values
        countries = u["countries"]

        print("\n results for line A, Table 4 and Table 5")
        res = load_data_and_calculate("pred_FIVB")
        print("\n results for line B, Table 4")
        res = load_data_and_calculate("pred_FIVB_xi_FIVB_eta_02")
        print("\n results for line C, Table 4")
        res = load_data_and_calculate("pred_FIVB_xi_FIVB_eta_02_r_tilde")
        print("\n results for line D, Table 4")
        res = load_data_and_calculate("pred_FIVB_xi_1_eta_02")
        print("\n results for line E, Table 4 and Table 5")
        res = load_data_and_calculate("pred_FIVB_xi_1_eta_02_r_tilde")
        print("\n results for line F, Table 4 and Table 5")
        res = load_data_and_calculate("pred_CL_xi_1_eta_02")
        # res = load_data_and_calculate("pred_CL_logistic")


if __name__ == "__main__":
    main()
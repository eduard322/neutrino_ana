import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, matthews_corrcoef, f1_score, accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import lightgbm
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, cross_validate, cross_val_score
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
import plotly.express as px


import optuna
import warnings
warnings.filterwarnings('ignore')

class ML_pipeline:
    def __init__(self, data_obj, preweighting = False, list_of_interest = None):
        print("Reading preprocessed dataset...")
        if preweighting:
            data_obj.read_files()
            data_obj.prim_convert_int()
            self.Data = data_obj.Data
        else:
            data_obj.read_parq()
            self.Data = data_obj.Data_conv
        print("Done!")
        print(self.Data["mu"].columns)
        self.list_of_interest = list_of_interest
        self.tau_decayed = data_obj.tau_decayed

    
    

    def construct_dataset(self, neutrals = [], clear_cluster = True, condition_on_dataset = None, smearing = False, IP = True):
        self.IP = IP
        print("Constructing dataset for classification...")
        print(f"Excluding following neutrals: {neutrals}")
        
        Data_mu = self.Data["mu"]
        #print(Data_mu["weight"])
        numu = Data_mu.query(f"id == 14 & m_id == -1")[["Event", "E", "px", "py", "pz"]]
        numu.columns = ["Event_nu", "E_nu", "px_nu", "py_nu", "pz_nu"]
        hardblob = Data_mu.query(f"id == 2000000002 & m_id != 5")[["Event", "E", "px", "py", "pz"]].groupby("Event").sum().reset_index()
        hardblob.columns = ["Event_hardblob", "E_hardblob", "px_hardblob", "py_hardblob", "pz_hardblob"]
        nucl = Data_mu.query(f"id == 1000260560 & m_id == -1")[["Event", "E", "px", "py", "pz"]].groupby("Event").sum().reset_index()
        nucl.columns = ["Event_nucl", "E_nucl", "px_nucl", "py_nucl", "pz_nucl"]
        muons = Data_mu.query(f"id == 13 & m_id == 0")
        if clear_cluster:
            hadrons = Data_mu.query(f"id == 2000000001")
        else:
            hadrons = Data_mu.loc[~Data_mu.name.isin(neutrals)].query("final == 1 & m_id != 0").drop(columns = ["name"]).groupby("Event").sum().reset_index()
        muons.columns = [col + "_mu" for col in muons.columns]
        hadrons.columns = [col + "_hadr" for col in hadrons.columns]
        muons = muons.merge(hadrons, how = "outer", left_on = "Event_mu", right_on = "Event_hadr")
        muons_mu_0 = muons.merge(numu, how = "outer", left_on = "Event_mu", right_on = "Event_nu")
        muons_mu_1 = muons_mu_0.merge(hardblob, how = "outer", left_on = "Event_mu", right_on = "Event_hardblob")
        muons_mu_2 = muons_mu_1.merge(nucl, how = "outer", left_on = "Event_mu", right_on = "Event_nucl")
        muons_mu_2.loc[:,"label"] = pd.Series(["numu"]*len(muons_mu_2["id_mu"]))

        #############
        if self.IP:
            out_IP = pd.read_csv("data/out_IP_mu_3cm.csv")
            out_IP = out_IP.drop(columns = ['Unnamed: 0'])
            muons_mu = muons_mu_2.merge(out_IP, how = "inner", left_on = "Event_mu", right_on = "Event_mu_IP")
            muons_mu["E_mu_old"] = muons_mu["E_mu"]
            muons_mu["px_mu_old"] = muons_mu["px_mu"]
            muons_mu["py_mu_old"] = muons_mu["py_mu"]
            muons_mu["pz_mu_old"] = muons_mu["pz_mu"]
            muons_mu["E_mu"] = muons_mu["E_mu_IP"]
            muons_mu["px_mu"] = muons_mu["px_mu_IP"]
            muons_mu["py_mu"] = muons_mu["py_mu_IP"]
            muons_mu["pz_mu"] = muons_mu["pz_mu_IP"]
        else:
            muons_mu = muons_mu_2
        #############
        # print(muons)
        print(muons_mu)
        # exit(0)
        fig, ax = plt.subplots(1,4, figsize = (20, 5))
        print(muons_mu["E_nu"] + muons_mu["E_nucl"] - muons_mu["E_hadr"] - muons_mu["E_mu"] - muons_mu["E_hardblob"])
        ax[0].hist(muons_mu["px_nu"] + muons_mu["px_nucl"] - muons_mu["px_hadr"] - muons_mu["px_mu"] - muons_mu["px_hardblob"], bins = 50)
        ax[1].hist(muons_mu["py_nu"] + muons_mu["py_nucl"] - muons_mu["py_hadr"] - muons_mu["py_mu"] - muons_mu["py_hardblob"], bins = 50)
        ax[2].hist(muons_mu["pz_nu"] + muons_mu["pz_nucl"] - muons_mu["pz_hadr"] - muons_mu["pz_mu"] - muons_mu["pz_hardblob"], bins = 50)
        ax[3].hist(muons_mu["E_nu"] + muons_mu["E_nucl"] - muons_mu["E_hadr"] - muons_mu["E_mu"] - muons_mu["E_hardblob"], bins = 50)
        ax[0].set_xlabel("px_before - px_after [GeV]")
        ax[1].set_xlabel("py_before - py_after [GeV]")
        ax[2].set_xlabel("pz_before - pz_after [GeV]")
        ax[3].set_xlabel("E_before - E_after [GeV]")
        fig.suptitle("pz > 0 & neutrinos excluded")
        for i in range(4):
            ax[i].set_yscale("log")
        
        #fig, ax = plt.subplots()
        #ax.hist(muons_mu["E_mu"], bins = 100, histtype = "step", label = "E_mu")
        #ax.hist(muons_mu["E_mu_IP"], bins = 100, histtype = "step", label = "E_mu_IP")
        #plt.show()


        Data_mu = self.Data["tau"]
        #print(Data_mu["weight"])
        # numu = Data_mu.query(f"id == 16 & m_id == -1")[["Event", "E"]]
        # numu.columns = ["Event_nu", "E_nu"]        
        numu = Data_mu.query(f"id == 16 & m_id == -1")[["Event", "E", "px", "py", "pz"]]
        numu.columns = ["Event_nu", "E_nu", "px_nu", "py_nu", "pz_nu"]
        hardblob = Data_mu.query(f"id == 2000000002")[["Event", "E", "px", "py", "pz"]].groupby("Event").sum().reset_index()
        hardblob.columns = ["Event_hardblob", "E_hardblob", "px_hardblob", "py_hardblob", "pz_hardblob"]
        nucl = Data_mu.query(f"id == 1000260560 & m_id == -1")[["Event", "E", "px", "py", "pz"]].groupby("Event").sum().reset_index()
        nucl.columns = ["Event_nucl", "E_nucl", "px_nucl", "py_nucl", "pz_nucl"]
        muons = Data_mu.query(f"id == 13 & m_id == {4 if self.tau_decayed else 0}")
        if clear_cluster:
            hadrons = Data_mu.query(f"id == 2000000001")
        else:
            hadrons = Data_mu.loc[~Data_mu.name.isin(neutrals)].query("final == 1 & m_id != 4").drop(columns = ["name"]).groupby("Event").sum().reset_index()
        muons.columns = [col + "_mu" for col in muons.columns]
        hadrons.columns = [col + "_hadr" for col in hadrons.columns]
        muons = muons.merge(hadrons, how = "inner", left_on = "Event_mu", right_on = "Event_hadr")
        muons_tau_0 = muons.merge(numu, how = "inner", left_on = "Event_mu", right_on = "Event_nu")
        muons_tau_1 = muons_tau_0.merge(hardblob, how = "inner", left_on = "Event_mu", right_on = "Event_hardblob")
        muons_tau_2 = muons_tau_1.merge(nucl, how = "inner", left_on = "Event_mu", right_on = "Event_nucl")
        muons_tau_2.loc[:,"label"] = pd.Series(["nutau"]*len(muons_tau_2["id_mu"]))
        #############
        if self.IP:
            out_IP = pd.read_csv("data/out_IP_tau_3cm.csv")
            out_IP = out_IP.drop(columns = ['Unnamed: 0'])
            print("TAU")
            print(out_IP)
            print(muons_tau_2)
            print("TAU")
            muons_tau = muons_tau_2.merge(out_IP, how = "inner", left_on = "Event_mu", right_on = "Event_mu_IP")
            muons_tau["E_mu_old"] = muons_tau["E_mu"]
            muons_tau["px_mu_old"] = muons_tau["px_mu"]
            muons_tau["py_mu_old"] = muons_tau["py_mu"]
            muons_tau["pz_mu_old"] = muons_tau["pz_mu"]
            muons_tau["E_mu"] = muons_tau["E_mu_IP"]
            muons_tau["px_mu"] = muons_tau["px_mu_IP"]
            muons_tau["py_mu"] = muons_tau["py_mu_IP"]
            muons_tau["pz_mu"] = muons_tau["pz_mu_IP"]
        else:
            muons_tau = muons_tau_2
        #############
        print("IP !!!!!!!!!")
        print(muons_mu)
        print(muons_tau)
        print("IP !!!!!!!!!")
        df_muon = pd.concat([muons_mu, muons_tau], axis = 0, ignore_index = True)
        #df_muon["diff"] = df_muon["E_nu"] + df_muon["E_nucl"] - df_muon["E_hadr"] - df_muon["E_mu"] - df_muon["E_hardblob"]
        #df_muon["diff_pz"] = df_muon["pz_nu"] + df_muon["pz_nucl"] - df_muon["pz_hadr"] - df_muon["pz_mu"] - df_muon["pz_hardblob"]
        #print(df_muon["weight_mu"])
        # fig, ax = plt.subplots(1,4)
        # ax[0].hist(muons_tau["px_nu"] - muons_tau["px_hadr"] - muons_tau["px_mu"], bins = 50)
        # ax[1].hist(muons_tau["py_nu"] - muons_tau["py_hadr"] - muons_tau["py_mu"], bins = 50)
        # ax[2].hist(muons_tau["pz_nu"] - muons_tau["pz_hadr"] - muons_tau["pz_mu"], bins = 50)
        # ax[3].hist(muons_tau["E_nu"] - muons_tau["E_hadr"] - muons_tau["E_mu"], bins = 50)
        # ax[0].set_xlabel("px_nu - px_hadr - px_mu [GeV]")
        # ax[1].set_xlabel("py_nu - py_hadr - py_mu [GeV]")
        # ax[2].set_xlabel("pz_nu - pz_hadr - pz_mu [GeV]")
        # ax[3].set_xlabel("E_nu - E_hadr - E_mu [GeV]")
        # plt.show()
        # # print(df_muon.columns)
        # # print(df_muon["label"])
        # exit(0)
        if smearing:
            # np.random.seed()
            df_muon["E_hadr"] = np.random.normal(df_muon["E_hadr"], 0.1*df_muon["E_hadr"])
            df_muon["E_mu"] = np.random.normal(df_muon["E_mu"], 0.15*df_muon["E_mu"])
            df_muon["px_mu"] = np.random.normal(df_muon["px_mu"], 0.09*np.abs(df_muon["px_mu"]))
            df_muon["py_mu"] = np.random.normal(df_muon["py_mu"], 0.09*np.abs(df_muon["py_mu"]))
            df_muon["px_hadr"] = np.random.normal(df_muon["px_hadr"], 0.06*np.abs(df_muon["px_hadr"]))
            df_muon["py_hadr"] = np.random.normal(df_muon["py_hadr"], 0.06*np.abs(df_muon["py_hadr"]))
        ###
        
        df_muon["px_miss"] = -df_muon["px_mu"].values - df_muon["px_hadr"].values
        df_muon["py_miss"] = -df_muon["py_mu"].values - df_muon["py_hadr"].values
        #df_muon["pz_miss"] = df_muon["P_nu"].values - df_muon["pz_mu"].values - df_muon["pz_hadr"].values
        #df_muon["P_miss"] = np.sqrt(df_muon["px_miss"]**2 + df_muon["py_miss"]**2 + df_muon["pz_miss"]**2)


        muon_sin = np.sqrt(df_muon["px_mu"].values**2 + df_muon["py_mu"].values**2)/np.sqrt(df_muon["px_mu"].values**2 + df_muon["py_mu"].values**2 + df_muon["pz_mu"].values**2)
        hadr_sin = np.sqrt(df_muon["px_hadr"].values**2 + df_muon["py_hadr"].values**2)/np.sqrt(df_muon["px_hadr"].values**2 + df_muon["py_hadr"].values**2 + df_muon["pz_hadr"].values**2)
        #df_muon["Pt_miss"] = np.abs(df_muon["E_hadr"].values*hadr_sin - df_muon["E_mu"].values*muon_sin)
        #df_muon["Pt_miss"] = df_muon["E_hadr"].values*hadr_sin - df_muon["E_mu"].values*muon_sin
        df_muon["Pt_hadr"] = df_muon["E_hadr"].values*hadr_sin
        df_muon["Pt_mu"] = df_muon["E_mu"].values*muon_sin


        df_muon["Pt_miss_old"] = np.sqrt(df_muon["px_miss"]**2 + df_muon["py_miss"]**2)
        #df_muon["Pt_miss"] = df_muon["Pt_miss_old"]
        df_muon["Pt_mu_old"] = np.sqrt(df_muon["px_mu"]**2 + df_muon["py_mu"]**2)
        df_muon["P_mu_old"] = np.sqrt(df_muon["px_mu"]**2 + df_muon["py_mu"]**2 + df_muon["pz_mu"]**2)
        df_muon["Pt_hadr_old"] = np.sqrt(df_muon["px_hadr"]**2 + df_muon["py_hadr"]**2)
        df_muon["P_hadr_old"] = np.sqrt(df_muon["px_hadr"]**2 + df_muon["py_hadr"]**2 + df_muon["pz_hadr"]**2)


        # df_muon["Pt_miss"] = np.abs(df_muon["E_hadr"].values*hadr_sin - df_muon["E_mu"].values*muon_sin)
        # df_muon["Pt_hadr"] = df_muon["E_hadr"].values*hadr_sin
        # df_muon["Pt_mu"] = df_muon["E_mu"].values*muon_sin

        # df_muon["Pt_miss"] = df_muon["Pt_miss_old"]
        # df_muon["Pt_hadr"] = df_muon["Pt_hadr_old"]
        # df_muon["Pt_mu"] = df_muon["Pt_mu_old"]

        df_muon["Pt_miss/E_mu"] = df_muon["Pt_miss"]/df_muon["E_mu"]
        df_muon["Pt_miss/E_hadr"] = df_muon["Pt_miss"]/df_muon["E_hadr"]
        df_muon["Pt_miss/Pt_mu"] = df_muon["Pt_miss"]/df_muon["Pt_mu"]
        df_muon["Pt_miss/Pt_hadr"] = df_muon["Pt_miss"]/df_muon["Pt_hadr"]
        df_muon["E_mu/E_hadr"] = df_muon["E_mu"]/df_muon["E_hadr"]
        df_muon["Pt_mu/Pt_hadr"] = df_muon["Pt_mu"]/df_muon["Pt_hadr"]
        df_muon["Pt_mu/E_hadr"] = df_muon["Pt_mu"]/df_muon["E_hadr"]
        df_muon["Pt_hadr/E_mu"] = df_muon["Pt_hadr"]/df_muon["E_mu"]

        df_muon["Pt_missxE_mu"] = df_muon["Pt_miss"]*df_muon["E_mu"]

        # df_muon["E_mu/Pt_miss"] = df_muon["E_mu"]/df_muon["Pt_miss"]
        # df_muon["E_hadr/Pt_miss"] = df_muon["E_hadr"]/df_muon["Pt_miss"]
        # df_muon["Pt_miss/Pt_mu"] = df_muon["Pt_miss"]/df_muon["Pt_mu"]
        # df_muon["Pt_hadr/Pt_miss"] = df_muon["Pt_hadr"]/df_muon["Pt_miss"]
        # df_muon["E_hadr/E_mu"] = df_muon["E_hadr"]/df_muon["E_mu"]
        # df_muon["Pt_hadr/Pt_mu"] = df_muon["Pt_hadr"]/df_muon["Pt_mu"]

        df_muon["anglePtmissandPtmuon"] = np.arccos((-df_muon["px_miss"].values*df_muon["px_mu"].values - df_muon["py_miss"].values*df_muon["py_mu"].values)/(df_muon["Pt_miss_old"].values*df_muon["Pt_mu_old"].values))
        df_muon["anglePtmissandPthadr"] = np.arccos((-df_muon["px_miss"]*df_muon["px_hadr"] - df_muon["py_miss"]*df_muon["py_hadr"])/(df_muon["Pt_miss_old"]*df_muon["Pt_hadr_old"]))
        df_muon["anglePtmuonandPthadr"] = np.arccos((-df_muon["px_mu"]*df_muon["px_hadr"] - df_muon["py_mu"]*df_muon["py_hadr"])/(df_muon["Pt_mu_old"]*df_muon["Pt_hadr_old"]))

        # df_muon["anglePtmissandPtmuon"] = np.arccos()


        df_muon["anglePtmissandPtmuon"] = 180*df_muon["anglePtmissandPtmuon"]/np.pi
        df_muon["anglePtmissandPthadr"] = 180*df_muon["anglePtmissandPthadr"]/np.pi
        df_muon["anglePtmuonandPthadr"] = 180*df_muon["anglePtmuonandPthadr"]/np.pi


        #df_muon["Pt_mu_on_Pt_miss"] = (df_muon["px_miss"]*df_muon["px_mu"] + df_muon["py_miss"]*df_muon["py_mu"])/df_muon["Pt_miss"]
        #df_muon["Pt_hadr_on_Pt_miss"] = (df_muon["px_miss"]*df_muon["px_hadr"] + df_muon["py_miss"]*df_muon["py_hadr"])/df_muon["Pt_miss"]
        p_muon = np.sqrt(df_muon["px_mu"]**2 + df_muon["py_mu"]**2 + df_muon["pz_mu"]**2)
        p_hadr = np.sqrt(df_muon["px_hadr"]**2 + df_muon["py_hadr"]**2 + df_muon["pz_hadr"]**2)
        df_muon["anglePhadrandPmuon"] = np.arccos((df_muon["px_hadr"]*df_muon["px_mu"] + df_muon["py_hadr"]*df_muon["py_mu"] + df_muon["pz_hadr"]*df_muon["pz_mu"])/(p_muon*p_hadr))

        df_muon["anglePhadrandPmuon"] = 180*df_muon["anglePhadrandPmuon"]/np.pi
        #print(df_muon)
        self.df_muon_1 = df_muon.dropna()
        #print(df_muon)
        self.num_numu = self.df_muon_1.query("label == 'numu'")["weight_mu"].sum()
        self.num_nutau = self.df_muon_1.query("label == 'nutau'")["weight_mu"].sum()
        print(f"!!!!!!!! {self.num_numu/self.num_nutau}")
        self.condition_on_dataset = condition_on_dataset
        if self.condition_on_dataset is None:
            self.nu_data = self.df_muon_1[["label", "E_nu"]]
        else:
            self.df_muon_1 = self.df_muon_1.query(self.condition_on_dataset)
            self.nu_data = self.df_muon_1[["label", "E_nu"]]
        self.num_numu_after = self.df_muon_1.query("label == 'numu'")["weight_mu"].sum()
        self.num_nutau_after = self.df_muon_1.query("label == 'nutau'")["weight_mu"].sum()
        #print(f"!!!!!!!! {self.num_numu_after/self.num_nutau_after}")
        excluded_list = ['Number_mu', 'id_mu', 'm_id_mu', 'final_mu', 'Xin_mu', 'Yin_mu', 'Zin_mu', 'Rin_mu',
            'Number_hadr', 'id_hadr', 'm_id_hadr', 'final_hadr', 'Xin_hadr', 'Yin_hadr', 'Zin_hadr', 'Rin_hadr',
            'weight_hadr', 'px_miss', 'py_miss', 'px_hadr', 'py_hadr', 'pz_hadr', 'P_in_hadr', 'P_hadr', 'P_in_mu', 
                        #'Pt_miss_old', 
                        'Pt_mu_old', 'P_mu_old', 'Pt_hadr_old', 'P_hadr_old', 'pz_mu', "E_nu", "Event_nu", 
                        "Event_mu", "Event_hadr", "name_mu", "name_hadr", "Event_mu_IP",
                        "E_mu_old", "px_mu_old", "py_mu_old", "pz_mu_old",
                        "E_mu_IP", "px_mu_IP", "py_mu_IP", "pz_mu_IP",
                        "Event_hardblob", "E_hardblob", "px_hardblob", "py_hardblob", "pz_hardblob",
                        "Event_nucl", "E_nucl", "px_nucl", "py_nucl", "pz_nucl",
                        "E_nu", "px_nu", "py_nu", "pz_nu"
                        
                        #'Pt_miss/Pt_hadr', 'Pt_miss/Pt_mu', 'Pt_miss/E_mu', 'Pt_miss/E_mu', 'Pt_mu/Pt_hadr', 
                        #'anglePtmuonandPthadr'
                        ]

        self.df_muon_1 = self.df_muon_1.loc[:, ~self.df_muon_1.columns.isin(excluded_list)]
        # print("Removing label and weights from list_of_interest...")
        # self.list_of_interest = self.list_of_interest.remove("label").remove("weight_mu")


        print("Done!") 
        print(f"Dataset has following features: {self.df_muon_1.columns}")       
        #print(f"Neutrino info: {self.nu_data}")

    def check_smearing(self, number_of_tries = 10):
        pt_mu = []
        pt_tau = []
        pt_mu_weight = []
        pt_tau_weight = []
        condition_on_dataset = "anglePhadrandPmuon < 15."
        for n in range(number_of_tries):
            self.construct_dataset(smearing=True, condition_on_dataset=condition_on_dataset)
            pt_mu.append(self.df_muon_1.query("label == 'numu'")["Pt_miss"])
            pt_tau.append(self.df_muon_1.query("label == 'nutau'")["Pt_miss"])
            pt_mu_weight.append(self.df_muon_1.query("label == 'numu'")["weight_mu"])
            pt_tau_weight.append(self.df_muon_1.query("label == 'nutau'")["weight_mu"])       
            print(n, pt_mu[-1].values.mean(), pt_tau[-1].values.mean())
        
        fig, ax = plt.subplots(figsize = (12,12), dpi = 100)
        difs = []
        difs_mu, difs_tau = [], []
        for n in range(number_of_tries):
            bins = np.logspace(np.log10(1e-2), np.log10(1e2), 200)
            #bins = np.linspace(0, 40, 200)

            h_mu = ax.hist(pt_mu[n], histtype = "step", 
                    #density = True, 
                    weights = pt_mu_weight[n],
                    bins = bins, label = f"numu_{n}", alpha=0.2)

            #ax.axvline(h_mu[1][np.argmax(h_mu[0])], color = "blue")
            h_tau = ax.hist(pt_tau[n], histtype = "step", 
                    #density = True, 
                    weights = pt_tau_weight[n],
                    bins = bins, label = f"nutau_{n}", alpha=0.2)
            diff_point = np.abs(h_tau[1][np.argmax(h_tau[0])] - h_mu[1][np.argmax(h_mu[0])])
            difs.append(diff_point)
            difs_mu.append(h_mu[1][np.argmax(h_mu[0])])
            difs_tau.append(h_tau[1][np.argmax(h_tau[0])])
            print(n, h_mu[1][np.argmax(h_mu[0])], h_tau[1][np.argmax(h_tau[0])])


        self.construct_dataset(smearing=False)
        Data_conv = self.df_muon_1
        h_mu = ax.hist(Data_conv.query("label == 'numu'")["Pt_miss"], histtype = "step", 
                #density = True, 
                #weights = Data_conv.query("label == 'numu'")["weight_mu"],
                bins = bins, label = "numu")
        #ax.axvline(h_mu[1][np.argmax(h_mu[0])], color = "blue")
        h_tau = ax.hist(Data_conv.query("label == 'nutau'")["Pt_miss"], histtype = "step", 
                #density = True, 
                #weights = Data_conv.query("label == 'nutau'")["weight_mu"],
                bins = bins, label = "nutau")  
        diff_point_1 = np.abs(h_tau[1][np.argmax(h_tau[0])] - h_mu[1][np.argmax(h_mu[0])])
        print(f"No smearing: {h_mu[1][np.argmax(h_mu[0])]}, {h_tau[1][np.argmax(h_tau[0])]}, dif: {h_tau[1][np.argmax(h_tau[0])] - h_mu[1][np.argmax(h_mu[0])]}")
        print(f"Smearing average: {np.array(difs_mu).mean()}, {np.array(difs_tau).mean()}, dif: {np.array(difs_tau).mean() - np.array(difs_mu).mean()}")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("Pt_miss")
        difs = np.array(difs)
        ax.set_title(f"Difference btw MPV of Pt_miss is: {difs.mean():.2f} +- {difs.std():.2f}. W/o smearing:  {diff_point_1:.2f}")
        ax.legend()
        fig.savefig("pics/smearing_pt_miss.pdf")
        plt.show()

    def feature_vis(self):
        plt.rcParams.update({'font.size': 14})
        print("Visualising the features...")
        print(f"Condition on dataset set: {self.condition_on_dataset}")
        list_of_interest = self.df_muon_1.drop(columns=['label', "weight_mu"]).columns
        # list_of_interest = ['Pt_miss']
        #bins = [np.linspace(0, 2000, 100), np.linspace(0, 4000, 100), np.linspace(0, 200, 100), np.linspace(0, 1., 100), np.linspace(0, 10., 100)]
        ranges = []
        # Data_conv = df_muon_1.loc[df_muon_1["Pt_miss/E_mu"] > 7.5]
        
        
        Data_conv = self.df_muon_1
        print(f"{Data_conv.columns} to be plotted...")
        nutau_events_number = len(Data_conv.query("label == 'nutau'")["E_mu"])
        weighted_nutau_events_number = Data_conv.query("label == 'nutau'")["weight_mu"].sum()/self.df_muon_1.query("label == 'nutau'")["weight_mu"].sum()
        print(f"Number of nutau events {nutau_events_number}, percentage: {weighted_nutau_events_number*100}%")

        bad_params = []
        good_params = []

        for param in list_of_interest:
            min_mu, min_tau = min(Data_conv.query("label == 'numu'")[param]), min(Data_conv.query("label == 'nutau'")[param])
            max_mu, max_tau = max(Data_conv.query("label == 'numu'")[param]), max(Data_conv.query("label == 'nutau'")[param])
            print(f"{param}. Range for numu: [{min_mu} {max_mu}]")
            print(f"{param}. Range for nutau: [{min_tau} {max_tau}]")
            if (min_tau > max_mu and max_tau > max_mu) or (min_tau < min_mu and max_tau < min_mu):
                print("alarm")
                
            if max_tau < max_mu and min_tau > min_mu:
                print(f"{param} is a bad parameter")
                bad_params.append(param)
            ranges.append(f"[{min(Data_conv[param]):.2f} {max(Data_conv[param]):.2f}]")
            #bins = np.logspace(np.log10(1e-5), np.log10(max(Data_conv[param])), 200)
            #
            fig, ax = plt.subplots(figsize = (6,6), dpi = 100)
            if min(Data_conv[param]) > 0:
                bins = np.logspace(np.log10(min(Data_conv[param])), np.log10(max(Data_conv[param])), 200)
                ax.set_xscale("log")
            else:
                bins = np.linspace(min(Data_conv[param]), max(Data_conv[param]), 200)

            h_mu = ax.hist(Data_conv.query("label == 'numu'")[param], histtype = "step", 
                    #density = True, 
                    #weights = Data_conv.query("label == 'numu'")["weight_mu"],
                    bins = bins, label = "$\\nu_{\mu}$")
            #ax.axvline(weighted_mode(Data_conv.query("label == 'numu'")[param], Data_conv.query("label == 'numu'")["weight_mu"])[0][0], color = "red")
            #ax.axvline(h_mu[1][np.argmax(h_mu[0])], color = "blue")
            
            h_tau = ax.hist(Data_conv.query("label == 'nutau'")[param], histtype = "step", 
                    #density = True, 
                    #weights = Data_conv.query("label == 'nutau'")["weight_mu"],
                    bins = bins, label = "$\\nu_{\\tau}$")  
        #     ax.set_xlim([13.1,1e3])
            #ax.axvline(weighted_mode(Data_conv.query("label == 'nutau'")[param], Data_conv.query("label == 'nutau'")["weight_mu"])[0][0], color = "blue")
            #ax.axvline(h_tau[1][np.argmax(h_tau[0])], color = "orange")
            av_point = (h_tau[1][np.argmax(h_tau[0])] + h_mu[1][np.argmax(h_mu[0])])/2
            diff_point = np.abs(h_tau[1][np.argmax(h_tau[0])] - h_mu[1][np.argmax(h_mu[0])])/av_point
            #print(f"average point: {(h_tau[1][np.argmax(h_tau[0])] + h_mu[1][np.argmax(h_mu[0])])/2}, difference: {diff_point*100}")
            #print(f"tau point: {h_tau[1][np.argmax(h_tau[0])]}")
            if diff_point*100 > 100:
                good_params.append(param)
            ax.set_yscale("log")
            ax.set_xlabel(param)
            #ax.set_xlabel("$P_{T}^{\mathrm{miss}}$ [GeV]")
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
            #ax.set_title(param + f". {diff_point*100:.0f} % difference btw MPV")
            ax.legend()
            fig.savefig(f"./pics/{'IP/' if self.IP else ''}{param.replace('/', '_') if '/' in param else param}_{'IP' if self.IP else ''}.pdf")
        


        corr_matrix = Data_conv.drop(columns = ["label", "weight_mu"]).corr(method='spearman')
        # fig, ax1 = plt.subplots(figsize = (8,8), dpi = 200)
        res = sns.clustermap(corr_matrix, method='weighted', cmap='coolwarm', figsize=(16, 16))
        plt.savefig('pics/corr_matrix_0.pdf', format='pdf')

        fig, ax = plt.subplots(1, 2, figsize = (12,6))
        subtract_mu = self.nu_data.query("label == 'numu'")["E_nu"] - Data_conv.query("label == 'numu'")["E_mu"] - Data_conv.query("label == 'numu'")["E_hadr"]
        subtract_tau = self.nu_data.query("label == 'nutau'")["E_nu"] - Data_conv.query("label == 'nutau'")["E_mu"] - Data_conv.query("label == 'nutau'")["E_hadr"]
        ax[0].hist(subtract_mu, bins = 50)
        ax[0].set_title("numu")
        ax[0].set_xlabel("E_nu - E_hadr - E_mu [GeV]")
        ax[1].hist(subtract_tau, bins = 50)
        ax[1].set_title("nutau")
        ax[1].set_xlabel("E_nu - E_hadr - E_mu [GeV]")
        fig.savefig("pics/subtraction.pdf")
        #plt.show()
        print("Done!")

    def normalization(self, norm_type = "z-score"):
        df_to_vis = self.df_muon_1.drop(columns=["weight_mu", "label"])
        self.df_muon_1[df_to_vis.columns] = StandardScaler().fit_transform(df_to_vis)
    def tsne_vis(self):
        self.normalization(self.df_muon_1)
        df_to_vis = self.df_muon_1.drop(columns=["weight_mu", "label"])
        #label = self.df_muon_1.drop(columns=['label', "weight_mu"]).columns
        # Extract embedding data
        embedding_data = df_to_vis.values
        print(df_to_vis.columns)
        #exit()
        # Initialize and fit t-SNE (with 2 components for 2D visualization)
        tsne_model = TSNE(n_components=2, random_state=42)
        tsne_output = tsne_model.fit_transform(embedding_data)
        #tsne_output = PCA(n_components=2).fit_transform(embedding_data)
        #umap_2d = umap.UMAP(n_components=2, init='random', random_state=42)
        #umap_embedding = umap_2d.fit_transform(embedding_data)
        # Add t-SNE result to the DataFrame (optional, for reference)
        self.df_muon_1['tsne_x'] = tsne_output[:, 0]
        self.df_muon_1['tsne_y'] = tsne_output[:, 1]
        #normalized_df['umap_embedding_x'] = umap_embedding[:, 0]
        #normalized_df['umap_embedding_y'] = umap_embedding[:, 1]
        fig = px.scatter(
            self.df_muon_1, 
            x='tsne_x', 
            y='tsne_y',  
            color='label', 
            height=800
        )
        fig.update_layout(
            plot_bgcolor='white', paper_bgcolor='white'
        )
        fig.write_image("pics/tsne.pdf")

        df_to_vis = self.df_muon_1.drop(columns=["weight_mu", "label"])
                #label = self.df_muon_1.drop(columns=['label', "weight_mu"]).columns
                # Extract embedding data
        embedding_data = df_to_vis.values
                #exit()
                # Initialize and fit t-SNE (with 2 components for 2D visualization)
        pca_model = PCA(n_components=20, random_state=42)
        pca_model_output = pca_model.fit_transform(embedding_data)
        explained_variance = pca_model.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
        plt.axhline(y=0.9, color='r', linestyle='--')  # Horizontal line at 90% explained variance
        plt.title('Cumulative Explained Variance')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.savefig("pics/pca_explained.pdf")
    def train_and_test(self, random_state = 13, optuna_on = False):


        def hyperparam_opt():
            study = optuna.create_study(direction="maximize")
            def objective(trial):
                def my_metric(y_true, y_pred, sample_weight):
                    conf_mat = confusion_matrix(y_true, y_pred, sample_weight = sample_weight)
                    return conf_mat[1][1]/(conf_mat[0][1] + conf_mat[1][0])
                # Determine the hyperperatemers and their value ranges
                learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1)
                num_leaves = trial.suggest_int("num_leaves", 2, 256)
                max_depth = trial.suggest_int("max_depth", -1, 50)
                min_child_samples = trial.suggest_int("min_child_samples", 5, 100)
                subsample = trial.suggest_float("subsample", 0.5, 1.0)
                colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
                n_estimators = trial.suggest_int("n_estimators", 100, 1000)
                min_split_gain = trial.suggest_int("min_split_gain", 0, 10)
                reg_lambda = trial.suggest_int("reg_lambda", 0, 10.)

                # Create and train the model
                model = LGBMClassifier(
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                max_depth=max_depth,
                min_child_samples=min_child_samples,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                n_estimators=n_estimators,
                min_split_gain=min_split_gain,
                reg_lambda=reg_lambda,
                random_state=42,
                verbose = -1
                )
                model.fit(X_train, y_train, sample_weight=w_train)
                
                # Evaluate model and return the metric
                y_pred = model.predict(X_test)
                accuracy = my_metric(y_test, y_pred, sample_weight=w_test)
                return accuracy     


            study.optimize(objective, n_trials=20)
            print("Best trial:")
            print(" Value: {}".format(study.best_trial.value))
            print(" Params: {}".format(study.best_trial.params))
            return study.best_trial.params
        print("Starting ML part...")
        # if condition_on_dataset is None:
        #     df_muon_2 = self.df_muon_1
        # else:   
        #     df_muon_2 = self.df_muon_1.query(condition_on_dataset)
        
        if self.IP:
            df_muon_2 = self.df_muon_1.sample(12726*2)
        else:
            df_muon_2 = self.df_muon_1.sample(12726*2)            
        print(f"Features used: {df_muon_2.columns}")
        print(f"Condition used: {self.condition_on_dataset}")
        print(f"Random state is {random_state}")
        
        nutau_events_number = len(df_muon_2.query("label == 'nutau'")["E_mu"])
        after_cuts = df_muon_2.query("label == 'nutau'")["weight_mu"].sum()
        before_cuts = self.num_nutau
        weighted_nutau_events_number = after_cuts/before_cuts

        numu_events_number = len(df_muon_2.query("label == 'numu'")["E_mu"])
        after_cuts_mu = df_muon_2.query("label == 'numu'")["weight_mu"].sum()
        before_cuts_mu = self.num_numu
        weighted_numu_events_number = after_cuts_mu/before_cuts_mu


        print(f"Number of nutau events {nutau_events_number}, percentage: {weighted_nutau_events_number*100}%, abs value: {after_cuts}, before cuts: {before_cuts}")

        print(f"Number of numu events {numu_events_number}, percentage: {weighted_numu_events_number*100}%, abs value: {after_cuts_mu}, before cuts: {before_cuts_mu}")


        X = df_muon_2.drop(columns=['label', 'weight_mu']).values
        X_pca = X
        # X_pca = np.hstack((X_pca, df_muon_2['weight_mu'].values.reshape(-1, 1)))
        
        for i, feat in enumerate(df_muon_2.drop(columns=['label', 'weight_mu']).columns):
            print(i, feat)

        weights_general = df_muon_2['weight_mu'].values
        #weights_general = np.array([1]*len(df_muon_2['weight_mu'].values))
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X_pca, df_muon_2["label"].map({'numu':0,'nutau':1}).values, weights_general,
            random_state=random_state,  
            shuffle=True,
            stratify=df_muon_2["label"].map({'numu':0,'nutau':1}).values
        )
        
        X_test_0 = pd.DataFrame(X_test)
        X_test_0.columns = df_muon_2.drop(columns=['label', 'weight_mu']).columns
        X_test_0["weight"] = pd.Series(w_test)
        X_test_0["label"] = pd.Series(y_test)
        nutau_ev = X_test_0.query("Pt_miss > 0.621")["weight"].sum()
        numu_ev = X_test_0.query("Pt_miss <= 0.621")["weight"].sum()
        print(nutau_ev/X_test_0.query("label == 1")["weight"].sum(), numu_ev/X_test_0.query("label == 0")["weight"].sum())
        # exit(0)

        # w_train = X_train[:,-1]
        # X_train = X_train[:,:-1]
        # w_test = X_test[:,-1]
        # X_test = X_test[:,:-1]


        best_params = hyperparam_opt()
        #exit(0)


        #params = {'learning_rate': 0.01035845058344329, 'num_leaves': 5, 'max_depth': 49, 'min_child_samples': 33, 'subsample': 0.8933995991090503, 'colsample_bytree': 0.7064945016496684, 'n_estimators': 989, 'min_split_gain': 8, 'reg_lambda': 6}
        params = {'learning_rate': 0.03325038678174303, 'num_leaves': 133, 'max_depth': 6, 'min_child_samples': 58, 'subsample': 0.766607522192533, 'colsample_bytree': 0.5985389375226084, 'n_estimators': 497, 'min_split_gain': 9, 'reg_lambda': 9}
        # Initialize the LightGBM classifier
        # model = LGBMClassifier(boosting_type = "gbdt", objective='binary', metric='binary_logloss', 
        #                     max_depth=-1, 
        #                     n_estimators=1, 
        #                     num_leaves = 31,
        #                     min_child_samples = 60,
        #                     min_split_gain = 4,
        #                     reg_lambda = 5.0,
        #                     verbose = -1
        #                     )
        model = LGBMClassifier(**best_params, verbose = -1)
        #model = DecisionTreeClassifier(max_depth=-1, num_leaves = 31)
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            
        )
        pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate ROC-AUC score
        roc_auc = roc_auc_score(y_test, pred_proba, sample_weight=w_test)
        print(f'ROC-AUC Score: {roc_auc:.2f}')

        # Plot histogram of predicted probabilities
        plt.figure(figsize=(12, 8), dpi=100)
        plt.hist(pred_proba[y_test == 1], bins=50, alpha=1.0, histtype = "step", label='nutau', weights=w_test[y_test == 1])
        plt.hist(pred_proba[y_test == 0], bins=50, alpha=1.0, histtype = "step", label='numu', weights=w_test[y_test == 0])
        plt.yscale("log")
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Histogram of Predicted Probabilities')
        plt.legend(loc='upper right')
        plt.grid(True)
        #plt.show()
        fig, ax = plt.subplots(dpi = 200)
        lightgbm.plot_tree(model, ax = ax)
        # lightgbm.create_tree_digraph(model)
        fig.savefig("pics/tree.pdf")
        #fig.show()
        y_pred = model.predict(X_test)
        print('Classification Report:\n', classification_report(y_test, y_pred, sample_weight = w_test))
        print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred, sample_weight = w_test))
        print(f"Fraction comparison nutau: {confusion_matrix(y_test, y_pred, sample_weight = w_test)[1][1]/X_test_0.query("label == 1")["weight"].sum()}, {X_test_0.query("label == 1")["weight"].sum()}")
        print(f"Fraction comparison numu: {confusion_matrix(y_test, y_pred, sample_weight = w_test)[0][0]/X_test_0.query("label == 0")["weight"].sum()}")
        #exit(0)
        # Bootstrap
        def bootstrap_metric(x, 
                            y,
                            w,
                            metric_fn,
                            samples_cnt = 200,
                            alpha = 0.05,
                            random_state = 42):
            
            np.random.seed(random_state)
            b_metric = np.zeros(samples_cnt)
            for it in range(samples_cnt):
                poses = np.random.choice(x.shape[0], size=x.shape[0], replace=True)
                
                x_boot = x[poses]
                y_boot = y[poses]
                w_boot = w[poses]
                
                m_val = metric_fn(x_boot, y_boot, w_boot)
                b_metric[it] = m_val
                if it%50 == 0:
                    print(it, "times...")
            
            return b_metric
        
        alpha = 0.05


        def my_metric(y_true, y_pred, sample_weight):
            conf_mat = confusion_matrix(y_true, y_pred, sample_weight = sample_weight)
            return conf_mat[1][1]/(conf_mat[0][1] + conf_mat[1][0])
        
        boot_mat_score = bootstrap_metric(y_test, y_pred, w_test, metric_fn=lambda x, y, w: f1_score(y_true=x, y_pred=y, sample_weight=w) , alpha = alpha)
        print("Model: {0}".format("BDT"), " \t f1-score: ", np.quantile(boot_mat_score, q=[alpha/2, 1 - alpha/2]))
        boot_mat_score = bootstrap_metric(y_test, y_pred, w_test, metric_fn=lambda x, y, w: precision_score(y_true=x, y_pred=y, sample_weight=w) , alpha = alpha)
        print("Model: {0}".format("BDT"), " \t precision-score: ", np.quantile(boot_mat_score, q=[alpha/2, 1 - alpha/2]))
        boot_mat_score = bootstrap_metric(y_test, y_pred, w_test, metric_fn=lambda x, y, w: recall_score(y_true=x, y_pred=y, sample_weight=w) , alpha = alpha)
        print("Model: {0}".format("BDT"), " \t recall-score: ", np.quantile(boot_mat_score, q=[alpha/2, 1 - alpha/2]))
        boot_mat_score = bootstrap_metric(y_test, y_pred, w_test, metric_fn=lambda x, y, w: my_metric(y_true=x, y_pred=y, sample_weight=w) , alpha = alpha)
        print("Model: {0}".format("BDT"), " \t signal_noise-score: ", np.quantile(boot_mat_score, q=[alpha/2, 1 - alpha/2]))
        



        #         # Run classifier with cross-validation and plot ROC curves
        # cv = StratifiedKFold(n_splits=6)

        # tprs = []
        # aucs = []
        # mean_fpr = np.linspace(0, 1, 100)

        # y = df_muon_2["label"].map({'numu':0,'nutau':1}).values

        # fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
        # plt.grid()

        # # Assuming weights is an array of sample weights corresponding to the samples in X
        # weights = df_muon_2['weight_mu'].values
        # model = LGBMClassifier(boosting_type = "gbdt", objective='binary', metric='binary_logloss', 
        #             max_depth=-1, 
        #             n_estimators=1, 
        #             num_leaves = 31
        #             )
        # for i, (train, test) in enumerate(cv.split(X, y)):
        #     model.fit(X[train], y[train], sample_weight=weights[train])
        #     viz = RocCurveDisplay.from_estimator(
        #         model,
        #         X[test],
        #         y[test],
        #         sample_weight=weights[test],  # Use sample weights for the ROC computation
        #         name="ROC fold {}".format(i),
        #         alpha=0.3,
        #         lw=1,
        #         ax=ax,
        #     )
        #     interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        #     interp_tpr[0] = 0.0
        #     tprs.append(interp_tpr)
        #     aucs.append(viz.roc_auc)

        # ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

        # mean_tpr = np.mean(tprs, axis=0)
        # mean_tpr[-1] = 1.0
        # mean_auc = auc(mean_fpr, mean_tpr)
        # std_auc = np.std(aucs)
        # ax.plot(
        #     mean_fpr,
        #     mean_tpr,
        #     color="b",
        #     label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        #     lw=2,
        #     alpha=0.8,
        # )

        # std_tpr = np.std(tprs, axis=0)
        # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        # ax.fill_between(
        #     mean_fpr,
        #     tprs_lower,
        #     tprs_upper,
        #     color="grey",
        #     alpha=0.2,
        #     label=r"$\pm$ 1 std. dev.",
        # )

        # ax.set(
        #     xlim=[-0.05, 1.05],
        #     ylim=[-0.05, 1.05],
        #     title="ROC-AUC metrics",
        # )
        # ax.legend(loc="lower right")
        #plt.show()
        print("Done!")
        
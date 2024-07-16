import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

class ML_pipeline:
    def __init__(self, data_obj, list_of_interest = None):
        print("Reading preprocessed dataset...")
        data_obj.read_parq()
        print("Done!")
        self.Data = data_obj.Data_conv
        print(self.Data["mu"].columns)
        self.list_of_interest = list_of_interest
    
    


    def construct_dataset(self, neutrals = [], smearing = False):
        print("Constructing dataset for classification...")
        print(f"Excluding following neutrals: {neutrals}")
        muon_data_numu = self.Data["mu"].query("final == 1 & m_id == 0 & abs(id) == 13").copy().set_index("Event").drop(columns = ["name"])
        #####
        #hadron_data_numu = self.Data["mu"].loc[~self.Data["mu"].name.isin(neutrals)].query("final == 1 & m_id != 0 & pz > 0").drop(columns = ["name"]).groupby("Event").sum()
        hadron_data_numu = self.Data["mu"].query("id == 2000000001").drop(columns = ["name"]).groupby("Event").sum()
        #####
        muon_data_numu.columns = [col + "_mu" for col in muon_data_numu.columns]
        hadron_data_numu.columns = [col + "_hadr" for col in hadron_data_numu.columns]
        nu_data_numu = pd.DataFrame(self.Data["mu"].query("m_id == -1 & abs(id) == 14")["pz"])
        nu_data_numu.columns = ["P_nu"]
        muon_data_numu = pd.concat([muon_data_numu, hadron_data_numu], axis = 1)
        muon_data_numu["label"] = pd.Series(len(muon_data_numu["id_mu"])*["numu"])
        muon_data_nutau = self.Data["tau"].query("final == 1 & m_id == 0 & abs(id) == 13").copy().set_index("Event").drop(columns = ["name"])
        #####
        #hadron_data_nutau = self.Data["tau"].loc[~self.Data["tau"].name.isin(neutrals)].query("final == 1 & m_id != 0 & pz > 0").drop(columns = ["name"]).groupby("Event").sum()
        hadron_data_nutau = self.Data["tau"].query("id == 2000000001").drop(columns = ["name"]).groupby("Event").sum()
        #####
        muon_data_nutau.columns = [col + "_mu" for col in muon_data_nutau.columns]
        hadron_data_nutau.columns = [col + "_hadr" for col in hadron_data_nutau.columns]
        nu_data_nutau = pd.DataFrame(self.Data["tau"].query("m_id == -1 & abs(id) == 16")["pz"])
        nu_data_nutau.columns = ["P_nu"]
        muon_data_nutau = pd.concat([muon_data_nutau, hadron_data_nutau], axis = 1)
        muon_data_nutau["label"] = pd.Series(len(muon_data_nutau["id_mu"])*["nutau"])
        df_muon = pd.concat([muon_data_numu, muon_data_nutau], ignore_index = True)
        df_muon["px_miss"] = -df_muon["px_mu"].values - df_muon["px_hadr"].values
        df_muon["py_miss"] = -df_muon["py_mu"].values - df_muon["py_hadr"].values
        #df_muon["pz_miss"] = df_muon["P_nu"].values - df_muon["pz_mu"].values - df_muon["pz_hadr"].values
        #df_muon["P_miss"] = np.sqrt(df_muon["px_miss"]**2 + df_muon["py_miss"]**2 + df_muon["pz_miss"]**2)
        ### smearing
        if smearing:
            df_muon["E_hadr"] = np.random.normal(df_muon["E_hadr"], 0.1*df_muon["E_hadr"])
            df_muon["E_mu"] = np.random.normal(df_muon["E_mu"], 0.15*df_muon["E_mu"])
        ###

        muon_sin = np.sqrt(df_muon["px_mu"].values**2 + df_muon["py_mu"].values**2)/np.sqrt(df_muon["px_mu"].values**2 + df_muon["py_mu"].values**2 + df_muon["pz_mu"].values**2)
        hadr_sin = np.sqrt(df_muon["px_hadr"].values**2 + df_muon["py_hadr"].values**2)/np.sqrt(df_muon["px_hadr"].values**2 + df_muon["py_hadr"].values**2 + df_muon["pz_hadr"].values**2)
        df_muon["Pt_miss"] = np.abs(df_muon["E_hadr"].values*hadr_sin - df_muon["E_mu"].values*muon_sin)
        df_muon["Pt_hadr"] = df_muon["E_hadr"].values*hadr_sin
        #df_muon["Pt_mu"] = df_muon["E_mu"].values*muon_sin


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

        # df_muon["Pt_miss/E_mu"] = df_muon["Pt_miss"]/df_muon["E_mu"]
        # df_muon["Pt_miss/E_hadr"] = df_muon["Pt_miss"]/df_muon["E_hadr"]
        # df_muon["Pt_miss/Pt_mu"] = df_muon["Pt_miss"]/df_muon["Pt_mu"]
        # df_muon["Pt_miss/Pt_hadr"] = df_muon["Pt_miss"]/df_muon["Pt_hadr"]
        # df_muon["E_mu/E_hadr"] = df_muon["E_mu"]/df_muon["E_hadr"]
        # df_muon["Pt_mu/Pt_hadr"] = df_muon["Pt_mu"]/df_muon["Pt_hadr"]
        # df_muon["Pt_mu/E_hadr"] = df_muon["Pt_mu"]/df_muon["E_hadr"]
        # df_muon["Pt_hadr/E_mu"] = df_muon["Pt_hadr"]/df_muon["E_mu"]

        #df_muon["Pt_missxE_mu"] = df_muon["Pt_miss"]*df_muon["E_mu"]

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
        print(df_muon)
        corrected_weight = df_muon.query("label == 'nutau'")["weight_mu"] / 5.
        df_muon.loc[:, "weight_mu"] = pd.Series(df_muon.query("label == 'numu'")["weight_mu"].to_list() + corrected_weight.to_list())

        df_muon_1 = df_muon.dropna()
        print(df_muon)
        excluded_list = ['Number_mu', 'id_mu', 'm_id_mu', 'final_mu', 'Xin_mu', 'Yin_mu', 'Zin_mu', 'Rin_mu',
            'Number_hadr', 'id_hadr', 'm_id_hadr', 'final_hadr', 'Xin_hadr', 'Yin_hadr', 'Zin_hadr', 'Rin_hadr',
            'weight_hadr', 'px_miss', 'py_miss', 'px_hadr', 'py_hadr', 'pz_hadr', 'P_in_hadr', 'P_hadr', 'P_in_mu', 
                        'Pt_miss_old', 'Pt_mu_old', 'P_mu_old', 'Pt_hadr_old', 'P_hadr_old', 'pz_mu', 
                        'Pt_miss/Pt_hadr', 'Pt_miss/Pt_mu', 'Pt_miss/E_mu', 'Pt_miss/E_mu', 'Pt_mu/Pt_hadr', 
                        #'anglePtmuonandPthadr'
                        ]

        self.df_muon_1 = df_muon_1.loc[:, ~df_muon_1.columns.isin(excluded_list)]
        # print("Removing label and weights from list_of_interest...")
        # self.list_of_interest = self.list_of_interest.remove("label").remove("weight_mu")
        print("Done!")        


    def feature_vis(self, condition_on_dataset = None):
        print("Visualising the features...")
        print(f"Condition on dataset set: {condition_on_dataset}")
        list_of_interest = self.df_muon_1.drop(columns=['label', "weight_mu"]).columns
        #list_of_interest = ['Pt_miss/Pt_mu', 'Pt_hadr/Pt_mu']
        #bins = [np.linspace(0, 2000, 100), np.linspace(0, 4000, 100), np.linspace(0, 200, 100), np.linspace(0, 1., 100), np.linspace(0, 10., 100)]
        ranges = []
        # Data_conv = df_muon_1.loc[df_muon_1["Pt_miss/E_mu"] > 7.5]
        if condition_on_dataset is None:
            Data_conv = self.df_muon_1
        else:
            Data_conv = self.df_muon_1.query(condition_on_dataset)
        print(f"{Data_conv.columns} to be plotted...")
        nutau_events_number = len(Data_conv.query("label == 'nutau'")["E_mu"])
        weighted_nutau_events_number = Data_conv.query("label == 'nutau'")["weight_mu"].sum()/self.df_muon_1.query("label == 'nutau'")["weight_mu"].sum()
        print(f"Number of nutau events {nutau_events_number}, percentage: {weighted_nutau_events_number*100}%")

        bad_params = []
        good_params = []

        for param in list_of_interest:
            min_mu, min_tau = min(Data_conv.query("label == 'numu'")[param]), min(Data_conv.query("label == 'nutau'")[param])
            max_mu, max_tau = max(Data_conv.query("label == 'numu'")[param]), max(Data_conv.query("label == 'nutau'")[param])
            #print(f"{param}. Range for numu: [{min_mu} {max_mu}]")
            #print(f"{param}. Range for nutau: [{min_tau} {max_tau}]")
            if (min_tau > max_mu and max_tau > max_mu) or (min_tau < min_mu and max_tau < min_mu):
                print("alarm")
                
            if max_tau < max_mu and min_tau > min_mu:
                print(f"{param} is a bad parameter")
                bad_params.append(param)
            ranges.append(f"[{min(Data_conv[param]):.2f} {max(Data_conv[param]):.2f}]")
            bins = np.logspace(np.log10(min(Data_conv[param])), np.log10(max(Data_conv[param])), 200)
            #bins = np.linspace(min(Data_conv[param]), max(Data_conv[param]), 200)
            fig, ax = plt.subplots(figsize = (6,6), dpi = 100)
            h_mu = ax.hist(Data_conv.query("label == 'numu'")[param], histtype = "step", 
                    #density = True, 
                    weights = Data_conv.query("label == 'numu'")["weight_mu"],
                    bins = bins, label = "numu")
            #ax.axvline(weighted_mode(Data_conv.query("label == 'numu'")[param], Data_conv.query("label == 'numu'")["weight_mu"])[0][0], color = "red")
            ax.axvline(h_mu[1][np.argmax(h_mu[0])], color = "blue")
            
            h_tau = ax.hist(Data_conv.query("label == 'nutau'")[param], histtype = "step", 
                    #density = True, 
                    weights = Data_conv.query("label == 'nutau'")["weight_mu"],
                    bins = bins, label = "nutau")  
        #     ax.set_xlim([13.1,1e3])
            #ax.axvline(weighted_mode(Data_conv.query("label == 'nutau'")[param], Data_conv.query("label == 'nutau'")["weight_mu"])[0][0], color = "blue")
            ax.axvline(h_tau[1][np.argmax(h_tau[0])], color = "orange")
            av_point = (h_tau[1][np.argmax(h_tau[0])] + h_mu[1][np.argmax(h_mu[0])])/2
            diff_point = np.abs(h_tau[1][np.argmax(h_tau[0])] - h_mu[1][np.argmax(h_mu[0])])/av_point
            #print(f"average point: {(h_tau[1][np.argmax(h_tau[0])] + h_mu[1][np.argmax(h_mu[0])])/2}, difference: {diff_point*100}")
            #print(f"tau point: {h_tau[1][np.argmax(h_tau[0])]}")
            if diff_point*100 > 100:
                good_params.append(param)
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_xlabel(param)
            ax.set_title(param + f". {diff_point*100:.0f} % difference btw MPV")
            ax.legend()
            fig.savefig(f"./pics/{param.replace('/', '_') if '/' in param else param}.pdf")
        


        corr_matrix = Data_conv.drop(columns = ["label", "weight_mu"]).corr(method='spearman')
        # fig, ax1 = plt.subplots(figsize = (8,8), dpi = 200)
        res = sns.clustermap(corr_matrix, method='weighted', cmap='coolwarm', figsize=(16, 16))
        plt.savefig('pics/corr_matrix_0.pdf', format='pdf')
        #plt.show()
        print("Done!")

    def train_and_test(self, condition_on_dataset = None, random_state = 13):
        print("Starting ML part...")
        if condition_on_dataset is None:
            df_muon_2 = self.df_muon_1
        else:   
            df_muon_2 = self.df_muon_1.query(condition_on_dataset)
        
        
        print(f"Features used: {df_muon_2.columns}")
        print(f"Condition used: {condition_on_dataset}")
        print(f"Random state is {random_state}")
        
        nutau_events_number = len(df_muon_2.query("label == 'nutau'")["E_mu"])
        after_cuts = df_muon_2.query("label == 'nutau'")["weight_mu"].sum()
        before_cuts = self.df_muon_1.query("label == 'nutau'")["weight_mu"].sum()
        weighted_nutau_events_number = df_muon_2.query("label == 'nutau'")["weight_mu"].sum()/self.df_muon_1.query("label == 'nutau'")["weight_mu"].sum()

        numu_events_number = len(df_muon_2.query("label == 'numu'")["E_mu"])
        after_cuts_mu = df_muon_2.query("label == 'numu'")["weight_mu"].sum()
        before_cuts_mu = self.df_muon_1.query("label == 'numu'")["weight_mu"].sum()
        weighted_numu_events_number = df_muon_2.query("label == 'numu'")["weight_mu"].sum()/self.df_muon_1.query("label == 'numu'")["weight_mu"].sum()


        print(f"Number of nutau events {nutau_events_number}, percentage: {weighted_nutau_events_number*100}%, abs value: {after_cuts}, before cuts: {before_cuts}")

        print(f"Number of numu events {numu_events_number}, percentage: {weighted_numu_events_number*100}%, abs value: {after_cuts_mu}, before cuts: {before_cuts_mu}")


        X = df_muon_2.drop(columns=['label', 'weight_mu']).values
        X_pca = X
        # X_pca = np.hstack((X_pca, df_muon_2['weight_mu'].values.reshape(-1, 1)))
        
        for i, feat in enumerate(df_muon_2.drop(columns=['label', 'weight_mu']).columns):
            print(i, feat)

        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X_pca, df_muon_2["label"].map({'numu':0,'nutau':1}).values, df_muon_2['weight_mu'].values,
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

        # Initialize the LightGBM classifier
        model = LGBMClassifier(boosting_type = "gbdt", objective='binary', metric='binary_logloss', 
                            max_depth=-1, 
                            n_estimators=1, 
                            num_leaves = 31
                            )
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
        plt.show()
        fig, ax = plt.subplots(dpi = 200)
        lightgbm.plot_tree(model, ax = ax)
        # lightgbm.create_tree_digraph(model)
        fig.savefig("pics/tree.pdf")
        fig.show()
        y_pred = model.predict(X_test)
        print('Classification Report:\n', classification_report(y_test, y_pred, sample_weight = w_test))
        print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred, sample_weight = w_test))
        print(f"Fraction comparison nutau: {confusion_matrix(y_test, y_pred, sample_weight = w_test)[1][1]/X_test_0.query("label == 1")["weight"].sum()}")
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
        



                # Run classifier with cross-validation and plot ROC curves
        cv = StratifiedKFold(n_splits=6)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        y = df_muon_2["label"].map({'numu':0,'nutau':1}).values

        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
        plt.grid()

        # Assuming weights is an array of sample weights corresponding to the samples in X
        weights = df_muon_2['weight_mu'].values
        model = LGBMClassifier(boosting_type = "gbdt", objective='binary', metric='binary_logloss', 
                    max_depth=-1, 
                    n_estimators=1, 
                    num_leaves = 31
                    )
        for i, (train, test) in enumerate(cv.split(X, y)):
            model.fit(X[train], y[train], sample_weight=weights[train])
            viz = RocCurveDisplay.from_estimator(
                model,
                X[test],
                y[test],
                sample_weight=weights[test],  # Use sample weights for the ROC computation
                name="ROC fold {}".format(i),
                alpha=0.3,
                lw=1,
                ax=ax,
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title="ROC-AUC metrics",
        )
        ax.legend(loc="lower right")
        plt.show()
        print("Done!")
        
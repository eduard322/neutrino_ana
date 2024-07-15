import pandas as pd
import numpy as np
from scipy import stats as st
import ROOT as r
import matplotlib.pyplot as plt
from array import array
import matplotlib as mpl
from functools import wraps
import time
import pyarrow.parquet as pq
import pyarrow as pa

m_tau = 1.777
m_mu = 0.106
m_numu = 0
m_nutau = 0
L_tot = 5672.
L_snd = 480.


class Read_and_Process_Raw_Files:
    def __init__(self, experiment = "ship", form = "normal", files = None):
        print("Initializing the read process...")
        self.experiment = experiment
        self.form = form
        self.files = files
        if self.files is not None:
            self.config_init = {key: filename for key,filename in zip(["mu", "tau"], [f"./raw_data/{file}" for file in self.files])}
        self.parq_title_mu = f"mu_weighted_E_{self.experiment}_fixed.parquet"
        self.parq_title_tau = f"tau_weighted_E_{self.experiment}_fixed.parquet"
        print("Done!")
    def timeit(func):
        @wraps(func)
        def timeit_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
            return result
        return timeit_wrapper
    @timeit
    def read_files(self, read_sep = "\s+"):
        self.Data = {}
        print("Reading raw GENIE output...")
        for file in self.config_init:
            self.Data[file] = pd.read_csv(self.config_init[file], header = None, sep = read_sep)
            print(self.Data[file])
            print(f"{file} was read...")
            self.Data[file].columns = ["Event", "Number", "id", "m_id", "name", "px", "py", "pz", "E", "final", "Xin", "Yin", "Zin"]
            self.Data[file]["Rin"] = np.sqrt(self.Data[file]["Xin"]**2 + self.Data[file]["Yin"]**2 + self.Data[file]["Zin"]**2)

    def write_parq(self):
        for part, file in zip(self.part_list, [self.parq_title_mu, self.parq_title_tau]):
            table = pa.Table.from_pandas(self.part_list[part])
            pq.write_table(table, f"./data/{file}")
    
    @timeit
    def read_parq(self):
        self.Data_conv = {}
        for part, file in zip(["mu", "tau"], [self.parq_title_mu, self.parq_title_tau]):
            self.Data_conv[part] = pq.read_table(f"./data/{file}").to_pandas()

    @timeit
    def prim_convert(self):
        
        def setdecay_df_single(data):
            
            tau_lepton = r.TLorentzVector(array('f', data))
            event = r.TGenPhaseSpace()
            event.SetDecay(tau_lepton, 3, np.asarray([m_mu, m_numu, m_nutau]))
            event.Generate()
            muon = event.GetDecay(0)
            return [muon.Vect()[i] for i in range(3)] + [muon.E()]


        def setdecay_df_single_3body(data):
            
            tau_lepton = r.TLorentzVector(array('f', data))
            event = r.TGenPhaseSpace()
            event.SetDecay(tau_lepton, 3, np.asarray([m_mu, m_numu, m_nutau]))
            event.Generate()
            return ([event.GetDecay(k).Vect()[i] for i in range(3)] + [event.GetDecay(k).E()] for k in range(3))

        def define_w(energy, weight_list, energy_list):
            low, high = 0, len(energy_list) - 1
            def binary_search(a, a_list, low, high):
                if low == high:
                    return low
                mid = low + (high - low) // 2
                if a > a_list[mid]:
                    return binary_search(a, a_list, mid+1, high)
                else:
                    return binary_search(a, a_list, low, mid)
            ind_out = binary_search(energy, energy_list, low, high)
            return weight_list[ind_out]
        
        mu_weight, tau_weight = pd.read_csv(f"./input_flux/{self.experiment}_numu_flux.data", header = None, sep = "\s+"), pd.read_csv(f"./input_flux/{self.experiment}_nutau_flux.data", header = None, sep = "\s+")
        self.part_list = {"mu": [], "tau": []} 
        weight_temp = 0
        columns_old = list(self.Data["mu"].columns[:]) + ["weight"]
        for index, row in self.Data["mu"].iterrows():
            if np.abs(row["id"]) == 14 and row["m_id"] == -1:
                weight_temp = define_w(row["E"], mu_weight[1].to_list(), mu_weight[0].to_list())

            self.part_list["mu"].append(row.to_list() + [weight_temp])
        self.part_list["mu"] = pd.DataFrame(self.part_list["mu"])
        self.part_list["mu"].columns = columns_old

        weight_temp = 0
        fig, ax = plt.subplots(figsize = (6,6), dpi = 150)
        e_dict = {"e_mu": [], "e_nu_mu": [], "e_nu_tau": []}
        for index, row in self.Data["tau"].iterrows():
            if np.abs(row["id"]) == 16 and row["m_id"] == -1:
                weight_temp = define_w(row["E"], tau_weight[1].to_list(), tau_weight[0].to_list())
            if np.abs(row["id"]) != 15:
                self.part_list["tau"].append(row.to_list() + [weight_temp])
            else:
                tau_list, nu_mu, nu_tau = setdecay_df_single_3body(row[["px", "py", "pz", "E"]])
                [px, py, pz, E] = tau_list
                if np.abs(tau_list[-1] + nu_mu[-1] + nu_tau[-1] - row["E"]) > row["E"]*0.10:
                    print(row["Event"], "!!!!!!", tau_list[-1] + nu_mu[-1] + nu_tau[-1], row["E"])
                e_dict["e_mu"].append(tau_list[-1]), e_dict["e_nu_mu"].append(nu_mu[-1]), e_dict["e_nu_tau"].append(nu_tau[-1])
                P = np.sqrt(px**2 + py**2 + pz**2)         
                old_part = row[["Event", "Number"]].to_list() + [13, row["m_id"], row["name"]] + tau_list + row[["final", "Xin", "Yin", "Zin"]].to_list() + [row["Rin"], weight_temp]
                self.part_list["tau"].append(old_part)
        self.part_list["tau"] = pd.DataFrame(self.part_list["tau"])
        self.part_list["tau"].columns = list(self.Data["tau"].columns) + ["weight"]
        
        for e_l in e_dict:
            ax.hist(e_dict[e_l], bins = 150, label = e_l)
        ax.legend()
        ax.set_xlabel("Energy [GeV]")
        ax.set_title("3-body decay kinematics of $\\tau \\rightarrow \mu\nu\nu$")
        fig.show()

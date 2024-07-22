from scripts import *
import matplotlib.pyplot as plt



read_obj = read_and_process.Read_and_Process_Raw_Files(experiment = "ship", 
                                                       form = "normal",
                                                       tau_decayed = True)


ml_pipe = ml_analysis.ML_pipeline(data_obj = read_obj, 
                                  list_of_interest = ["E_hadr"])


# neutrals = ["pi0", "neutron", "gamma", 'K0', 'K0_bar', 'K_L0', 'Lambda0', 'Lambda0_bar', "antineutron", 'nu_e_bar']
#neutrals = ["neutron", "antineutron", 'nu_e_bar', 'K0', 'K0_bar', 'K_L0', 'HardBlob']

neutrals = []


ml_pipe.construct_dataset(neutrals = neutrals, smearing = False)

cond = "Event == 2"
Data_mu = ml_pipe.Data["mu"]
numu = Data_mu.query(f"id == 14 & m_id == -1 & {cond}")
muons = Data_mu.query(f"id == 13 & m_id == 0 & {cond}")
hadrons = Data_mu.query(f"id == 2000000001 & {cond}")
print(f"numu:\n {numu} \n muons:\n {muons} \n hadrons:\n {hadrons}")
subtract = numu['E'].values - muons['E'].values - hadrons['E'].values
print(f"Subtract:\n {subtract}")
fig, ax = plt.subplots()
#ax.hist(subtract, bins = 50, histtype = "step")
#plt.show()


cond = "Event >= 0"
Data_mu = ml_pipe.Data["tau"].query(cond)
nutau = Data_mu.query(f"id == 16 & m_id == -1")
muons = Data_mu.query(f"id == 13 & m_id == 4")
hadrons = Data_mu.query(f"id == 2000000001")
taus = Data_mu.query(f"id == 15 & m_id == 0")
print(f"numu:\n {nutau} \n muons:\n {muons} \n hadrons:\n {hadrons}, \n taus:\n {taus}")
# subtract = nutau['E'].values - muons['E'].values - hadrons['E'].values
# subtract_tau = nutau['E'].values - taus['E'].values - hadrons['E'].values
# print(f"Subtract:\n mu: {subtract}\n tau: {subtract_tau}")
# fig, ax = plt.subplots()
# ax.hist(subtract_tau, bins = 50, histtype = "step")
# plt.show()

# print(len(set(nutau["Event"].to_list())), len(set(muons["Event"].to_list())))
# print(set(nutau["Event"].to_list()).difference(set(muons["Event"].to_list())))


black_list = set(nutau["Event"].to_list()).difference(set(muons["Event"].to_list()))


cond = "Event >= 0"
Data_mu = ml_pipe.Data["tau"].query(cond)
Data_mu = Data_mu.loc[~Data_mu["Event"].isin(black_list)]
nutau = Data_mu.query(f"id == 16 & m_id == -1")
muons = Data_mu.query(f"id == 13 & m_id == 4")
hadrons = Data_mu.query(f"id == 2000000001")
taus = Data_mu.query(f"id == 15 & m_id == 0")
print(f"numu:\n {nutau} \n muons:\n {muons} \n hadrons:\n {hadrons}, \n taus:\n {taus}")
subtract = nutau['E'].values - muons['E'].values - hadrons['E'].values
subtract_tau = nutau['E'].values - taus['E'].values - hadrons['E'].values
print(f"Subtract:\n mu: {subtract}\n tau: {subtract_tau}")
fig, ax = plt.subplots()
ax.hist(subtract, bins = 50, histtype = "step")
plt.show()
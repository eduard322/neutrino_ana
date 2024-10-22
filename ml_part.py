from scripts import *
import argparse



list_of_interest = ['px_mu', 'py_mu', 'pz_mu', 'E_mu', 'weight_mu', 'E_hadr', 'label',
       'Pt_miss', 'Pt_hadr', 'Pt_mu', 'Pt_miss/E_mu', 'Pt_miss/E_hadr',
       'Pt_miss/Pt_mu', 'Pt_miss/Pt_hadr', 'E_mu/E_hadr', 'Pt_mu/Pt_hadr',
       'Pt_mu/E_hadr', 'Pt_hadr/E_mu', 'Pt_missxE_mu', 'anglePtmissandPtmuon',
       'anglePtmissandPthadr', 'anglePtmuonandPthadr', 'anglePhadrandPmuon']

files_ship = ["output_nu_mu_100k_fe_taudecayed_charm_cc", "output_nu_tau_100k_fe_taudecayed_charm_cc"]
#files_ship = ["mu_weighted_E_ship_fixed", "tau_weighted_E_ship_fixed"]
files_ship = ["out_nu_mu_100k_fe_taudecayed_charm_cc_ana", "out_nu_tau_100k_fe_charm_cc_ana"]
parser = argparse.ArgumentParser()


parser.add_argument('--experiment', dest='experiment', 
                    default="ship",
                    help='ship or advsnd')

args = parser.parse_args()

read_obj = read_and_process.Read_and_Process_Raw_Files(experiment = args.experiment, 
                                                       files = files_ship,
                                                       form = "normal",
                                                       tau_decayed = False)


ml_pipe = ml_analysis.ML_pipeline(data_obj = read_obj, 
                                  list_of_interest = list_of_interest)


# neutrals = ["pi0", "neutron", "gamma", 'K0', 'K0_bar', 'K_L0', 'Lambda0', 'Lambda0_bar', "antineutron", 'nu_e_bar']
neutrals = [
    #"neutron", "antineutron", "proton", "antiproton", 
            'nu_e_bar', 'nu_e', 'nu_mu_bar', 'nu_mu', 'nu_tau', 'nu_tau_bar', 'mu']

#neutrals = []


#condition_on_dataset = "anglePhadrandPmuon < 15. & Pt_miss > 0.75"
#condition_on_dataset = "Pt_miss/Pt_mu > 0.8"
# condition_on_dataset = "Pt_miss > 0.5"
condition_on_dataset = None
# condition_on_dataset = "IP > 5e-2"
ml_pipe.construct_dataset(neutrals = neutrals, clear_cluster = False, 
                          condition_on_dataset = condition_on_dataset, 
                          smearing = False,
                          IP = False)



#condition_on_dataset = None


#ml_pipe.check_smearing(number_of_tries=10)
#ml_pipe.feature_vis()
#ml_pipe.tsne_vis()
ml_pipe.train_and_test(random_state = 13, optuna_on = True)







from scripts import *
import argparse



list_of_interest = ['px_mu', 'py_mu', 'pz_mu', 'E_mu', 'weight_mu', 'E_hadr', 'label',
       'Pt_miss', 'Pt_hadr', 'Pt_mu', 'Pt_miss/E_mu', 'Pt_miss/E_hadr',
       'Pt_miss/Pt_mu', 'Pt_miss/Pt_hadr', 'E_mu/E_hadr', 'Pt_mu/Pt_hadr',
       'Pt_mu/E_hadr', 'Pt_hadr/E_mu', 'Pt_missxE_mu', 'anglePtmissandPtmuon',
       'anglePtmissandPthadr', 'anglePtmuonandPthadr', 'anglePhadrandPmuon']

parser = argparse.ArgumentParser()


parser.add_argument('--experiment', dest='experiment', 
                    default="ship",
                    help='ship or advsnd')

args = parser.parse_args()

read_obj = read_and_process.Read_and_Process_Raw_Files(experiment = args.experiment, 
                                                       form = "normal")


ml_pipe = ml_analysis.ML_pipeline(data_obj = read_obj, 
                                  list_of_interest = list_of_interest)


#neutrals = ["pi0", "neutron", "gamma", 'K0', 'K0_bar', 'K_L0', 'Lambda0', 'Lambda0_bar', "antineutron", 'nu_e_bar']
#neutrals = ["neutron", "antineutron", 'nu_e_bar', 'K0', 'K0_bar', 'K_L0', 'HardBlob']

neutrals = []

ml_pipe.construct_dataset(neutrals = neutrals)


condition_on_dataset = "anglePhadrandPmuon < 15."

ml_pipe.feature_vis(condition_on_dataset = condition_on_dataset)
ml_pipe.train_and_test(condition_on_dataset = condition_on_dataset, random_state = 13)







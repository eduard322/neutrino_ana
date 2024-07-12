from scripts import *




files = ["100k_nu_mu_converted_E_Fe", "100k_nu_tau_converted_E_Fe"]
#files = ["nu_mu_100k_W_advsnd_conv_precise", "nu_tau_100k_W_advsnd_conv_precise"]
experiment = "ship"

read_obj = read_and_process.Read_and_Process_Raw_Files(experiment = experiment, 
                                                       form = "normal", 
                                                       files = files)





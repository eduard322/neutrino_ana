from scripts import *
import argparse




files_ship = ["100k_nu_mu_converted_E_Fe", "100k_nu_tau_converted_E_Fe_taudecayed"]
files_advsnd = ["nu_mu_100k_W_advsnd_conv_precise", "nu_tau_100k_W_advsnd_conv_precise_taudecayed"]


parser = argparse.ArgumentParser()


parser.add_argument('--experiment', dest='experiment', 
                    default="ship",
                    help='ship or advsnd')

args = parser.parse_args()

read_obj = read_and_process.Read_and_Process_Raw_Files(experiment = args.experiment, 
                                                       form = "normal", 
                                                       files = files_ship if args.experiment == "ship" else files_advsnd,
                                                       tau_decayed = True)

read_obj.read_files()
read_obj.prim_convert()
read_obj.write_parq()



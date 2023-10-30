import argparse
import healpy as hp
from bbmaster.utils import PipelineManager
import numpy as np
import subprocess

def filter_map(man, schedule, bands, telescope, sample_rate, thinfp, scan_map, out_dir, group_size):
    # make the output dir
    subprocess.call(['mkdir','-p',out_dir])
    # prepare the command
    command = 'OMP_NUM_THREADS=4 mpiexec -n 8 toast_so_sim.py --schedule %s --bands %s --telescope %s --sample_rate %i --thinfp %i --sim_ground.hwp_angle hwp_angle --sim_ground.hwp_rpm 120 --sim_ground.median_weather --weights_azel.single_precision --corotate_lat.disable --polyfilter1D.disable --mapmaker.disable --demodulate.enable --demodulate.purge --sim_noise.disable --sim_atmosphere_coarse.disable --sim_atmosphere.disable --scan_map.enable --scan_map.file %s --filterbin.enable --filterbin.no_write_cov --filterbin.no_write_rcond --filterbin.no_write_hits --filterbin.write_hdf5 --pixels_healpix_radec.enable --pixels_healpix_radec.nside %i --out %s --job_group_size %i | tee "%s/log" ' % (schedule, bands, telescope, sample_rate, thinfp, scan_map, man.nside, out_dir, group_size, out_dir)
    print(command)
    #process = subprocess.Popen(command, shell=True, universal_newlines=True, stdin=PIPE, stdout=PIPE, stderr=PIPE )
    #output, error = process.communicate();
    return ;

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='toast filterer')
    parser.add_argument("--globals", type=str, help='Path to yaml with global parameters')
    parser.add_argument("--first-sim", type=int, help='Index of first sim')
    parser.add_argument("--num-sims", type=int, help='Number of sims')
    parser.add_argument("--sim-sorter", type=str, help='Name of sorting routine')
    parser.add_argument("--schedule", type=str, help='File with toast schedule')
    parser.add_argument("--bands", type=str, default='SAT_f090', help='Bands to run in toast, e.g. SAT_f090')
    parser.add_argument("--telescope", type=str, default='SAT1', help='Telescope to simulate e.g. SAT1')
    parser.add_argument("--sample-rate", type=int, default=40, help='Sampling rate of telescope')
    parser.add_argument("--thinfp", type=int, default=8, help='Thin the focal plane by how much?')
    parser.add_argument("--group-size", type=int, default=1, help='Group size')
    parser.add_argument("--output-dir", type=str, help='Output directory')
    
    o = parser.parse_args()
    man = PipelineManager(o.globals)

    sorter = getattr(man, o.sim_sorter)
    file_input_list = sorter(o.first_sim, o.num_sims, o.output_dir, which='input')
    file_output_list = sorter(o.first_sim, o.num_sims, o.output_dir, which='filtered')
    
    for fin, fout in zip(file_input_list, file_output_list):
        outdir = fout.replace('.fits','/')
        filter_map(man, o.schedule, o.bands, o.telescope, o.sample_rate, o.thinfp, fin, outdir, o.group_size)
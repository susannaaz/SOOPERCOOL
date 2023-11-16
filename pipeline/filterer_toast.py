import argparse
import healpy as hp
from bbmaster.utils import PipelineManager
import numpy as np
import subprocess
from bbmaster.utils_toast import *
import os

def run_bash_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Command output:", result.stdout)
        print("Command error (if any):", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Error running command:", e)
        
def make_schedule(sched_par, sched_patches):
    """
    Create schedule
    TODO: Currently files
    @schedules/schedule_sat.par \
    @schedules/patches_sat.txt --debug
    are already stored. set to read from global yaml.

    default out: schedules/schedule_sat_1day.txt
    """
    command = f'toast_ground_schedule @{sched_par} @{sched_patches} --debug'
    run_bash_command(command)

def filter_map(inputmap_fl, nside_out=nside, sample_rate = 32., sched_path='data/schedules/schedule_sat_1day.txt', add_noise=False):
    """
    TODO: define input map file
    TODO: read nside from yaml
    TODO: define stuff in yaml
    """
    # Initialize schedule
    schedule = toast.schedule.GroundSchedule()
    if not os.path.exists(sched_path):
        # TODO change to: if sched_path==None
        # Make schedule
        make_schedule('data/schedules/schedule_sat.par', 'data/schedules/patches_sat.txt')
    # Read schedule
    schedule.read(sched_path)
    # Setup focal plane
    focalplane = sotoast.SOFocalplane(hwfile=None,
                                      telescope=schedule.telescope_name,
                                      sample_rate=sample_rate * u.Hz,
                                      bands='SAT_f090',
                                      wafer_slots='w25', 
                                      tube_slots=None,
                                      thinfp=None,
                                      comm=None)
    # Setup telescope
    telescope = toast.Telescope(name=schedule.telescope_name,
                                focalplane=focalplane, 
                                site=toast.GroundSite("Atacama", schedule.site_lat,
                                                      schedule.site_lon, schedule.site_alt))
    # Create data object
    data = toast.Data()

    # Apply filters
    _, sim_gnd = apply_scanning(data, telescope, schedule) # HWP info in here, + all sim_ground stuff
    data, det_pointing_radec = apply_det_pointing_radec(data, sim_gnd)
    data, det_pointing_azel = apply_det_pointing_azel(data, sim_gnd)
    data, pixels_radec = apply_pixels_radec(data, det_pointing_radec, nside_out)
    data, weights_radec = apply_weights_radec(data, det_pointing_radec)
    if add_noise:
        _, noise_model = apply_noise_model(data)
        data, sim_noise = apply_sim_noise(data)
    
    # Input map
    IQUmap = hp.read_map(inputmap_fl, field=[0,1,2])

    # Scan map
    data, scan_map = apply_scan_map(data, inputmap_fl, pixels_radec, weights_radec)


#def filter_map(man, schedule, bands, telescope, sample_rate, thinfp, scan_map, out_dir, group_size):
#    # make the output dir
#    subprocess.call(['mkdir','-p',out_dir])
#    # prepare the command
#    command = 'srun -n $ntask -c $ncore --cpu_bind=cores toast_so_sim.py --schedule %s --bands %s --telescope %s --sample_rate %i --thinfp %i --sim_ground.hwp_angle hwp_angle --sim_ground.hwp_rpm 120 --sim_ground.median_weather --weights_azel.single_precision --corotate_lat.disable --polyfilter1D.disable --mapmaker.disable --demodulate.enable --demodulate.purge --sim_noise.disable --sim_atmosphere_coarse.disable --sim_atmosphere.disable --scan_map.enable --scan_map.file %s --filterbin.enable --filterbin.no_write_cov --filterbin.no_write_rcond --filterbin.no_write_hits --filterbin.write_hdf5 --pixels_healpix_radec.enable --pixels_healpix_radec.nside %i --out %s --job_group_size %i | tee "%s/log" ' % (schedule, bands, telescope, sample_rate, thinfp, scan_map, man.nside, out_dir, group_size, out_dir)
#    print(command)
#    #process = subprocess.Popen(command, shell=True, universal_newlines=True, stdin=PIPE, stdout=PIPE, stderr=PIPE )
#    #output, error = process.communicate();
#    return ;

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
    
    #for fin, fout in zip(file_input_list, file_output_list):
    #    outdir = fout.replace('.fits','/')
    #    filter_map(man, o.schedule, o.bands, o.telescope, o.sample_rate, o.thinfp, fin, outdir, o.group_size)

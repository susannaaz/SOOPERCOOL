import argparse
import healpy as hp
from bbmaster.utils import PipelineManager
import numpy as np
import subprocess
from bbmaster.utils_toast import *
import os

def filter_map(man, fin, outdir, sched_path, thinfp, instrument, band, sample_rate, add_noise, 
               query=1, res=0.02*coords.DEG):
            
    """
    TOAST-based filtering - adapting the toast_so_sim workflow from toast into BBMASTER, 
    incorporated as one of the steps in the pipeline.
    """
    import time
    start_time = time.time()

    nside_out = man.nside

    # Initialize schedule
    schedule = toast.schedule.GroundSchedule()
    
    if not os.path.exists(sched_path):
        raise FileNotFoundError(f"The corresponding schedule file {sched_path} is not stored.")
    
    # Read schedule
    print('Read schedule')
    schedule.read(sched_path)

    # Setup focal plane
    print('Initialize focal plane and telescope')
    focalplane = sotoast.SOFocalplane(hwfile=None,
                                      telescope=instrument, #schedule.telescope_name,
                                      sample_rate=sample_rate * u.Hz,
                                      bands=band,
                                      wafer_slots='w25', 
                                      tube_slots=None,
                                      thinfp=thinfp,
                                      comm=None)
    # Setup telescope
    telescope = toast.Telescope(name=instrument, #schedule.telescope_name,
                                focalplane=focalplane, 
                                site=toast.GroundSite("Atacama", schedule.site_lat,
                                                      schedule.site_lon, schedule.site_alt))
    # Create data object
    data = toast.Data()

    # Apply filters
    print('Apply filters')
    _, sim_gnd = apply_scanning(data, telescope, schedule) # HWP info in here, + all sim_ground stuff
    data, det_pointing_radec = apply_det_pointing_radec(data, sim_gnd)
    data, pixels_radec = apply_pixels_radec(data, det_pointing_radec, nside_out)
    data, weights_radec = apply_weights_radec(data, det_pointing_radec)
    if add_noise:
        _, noise_model = apply_noise_model(data)
        data, sim_noise = apply_sim_noise(data)
    
    # Scan map
    print('Scan input map')
    data, scan_map = apply_scan_map(data, fin, pixels_radec, weights_radec)
    
    # TODO: Temporary data products  
    # Save HDF5
    print('Save hdf5 and context files')
    hdf5_path = outdir
    if not os.path.isdir(hdf5_path):
        os.system('mkdir -p ' + hdf5_path)
        save_hdf5 = toast.ops.SaveHDF5(name="save_hdf5")
        save_hdf5.volume = hdf5_path
        save_hdf5.apply(data)
    #
    # Save context 
    import write_context
    export_dirs = [f"{hdf5_path}/"]  # Directory to search for HDF data files
    context_dir = f"{export_dirs[0]}"  # Change this to desired output context directory
    if not os.path.isfile(f'{context_dir}context.yaml'):
        os.system('mkdir -p ' + context_dir)
        write_context.create_context(context_dir, export_dirs) #TODO: currently exportdir and contextdir are the same 
    
    # Load context 
    context = Context(f'{context_dir}context.yaml')
    obs = context.obsdb.get()
    ids = context.obsdb.query(query)['obs_id']
    wafer_list = obs['wafer_slots']
    dets_dict = {'dets:wafer_slot':wafer_list}
    
    # Coadd observations for one wafer at 1 freq 
    # TODO: Curently using sotodlib filter-bin map-maker
    # TODO: might want to coadd based on other params 
    print('Create coadded maps')
    npol = 3
    npix = hp.nside2npix(nside_out) 
    coadd_map = np.zeros([npol, npix])
    coadd_weighted_map = np.zeros([npol, npix])
    coadd_weight = np.zeros([npol, npol, npix])
    for ind in range(len(ids)):
        # Load data
        obs_id = ids[ind]
        detsets = context.obsfiledb.get_detsets(obs_id) #Detsets correspond to separate files, i.e. treat as separate TODs.
        dets_dict['band'] = 'f090'
        meta = context.get_meta(obs_id=obs_id, dets=dets_dict)
        aman = context.get_obs(obs_id=obs_id, meta=meta)
        # Pre-process data (additional filters!)
        aman, proc_aman = preprocess(aman)
        # Make CAR2healpix maps
        map_dict = demod.make_map(aman, res=res, dsT=aman.dsT, demodQ=aman.demodQ, demodU=aman.demodU)
        m_hp = map_dict['map'].to_healpix(nside=nside_out, order=0)
        w_hp = map_dict['weight'].to_healpix(nside=nside_out, order=0)
        mt_hp = map_dict['weighted_map'].to_healpix(nside=nside_out, order=0)
        # Coadd maps 
        coadd_map += m_hp
        coadd_weighted_map += mt_hp
        coadd_weight += w_hp
    print((time.time() - start_time)/60, 'minutes')
    return coadd_map, coadd_weighted_map, coadd_weight
    
def filter_map_temp(man, schedule, bands, telescope, sample_rate, thinfp, scan_map, out_dir, group_size):
    # make the output dir
    subprocess.call(['mkdir','-p',out_dir])
    # prepare the command
    command = 'srun -n $ntask -c $ncore --cpu_bind=cores toast_so_sim.py --schedule %s --bands %s --telescope %s --sample_rate %i --thinfp %i --sim_ground.hwp_angle hwp_angle --sim_ground.hwp_rpm 120 --sim_ground.median_weather --weights_azel.single_precision --corotate_lat.disable --polyfilter1D.disable --mapmaker.disable --demodulate.enable --demodulate.purge --sim_noise.disable --sim_atmosphere_coarse.disable --sim_atmosphere.disable --scan_map.enable --scan_map.file %s --filterbin.enable --filterbin.no_write_cov --filterbin.no_write_rcond --filterbin.no_write_hits --filterbin.write_hdf5 --pixels_healpix_radec.enable --pixels_healpix_radec.nside %i --out %s --job_group_size %i | tee "%s/log" ' % (schedule, bands, telescope, sample_rate, thinfp, scan_map, man.nside, out_dir, group_size, out_dir)
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
    parser.add_argument("--schedule", type=str, default='data/schedules/schedule_sat_5hr.txt', help='File with toast schedule')
    parser.add_argument("--bands", type=str, default='SAT_f090', help='Bands to run in toast, e.g. SAT_f090')
    parser.add_argument("--telescope", type=str, default='SAT1', help='Telescope to simulate e.g. SAT1')
    parser.add_argument("--sample-rate", type=int, default=32, help='Sampling rate of telescope')
    parser.add_argument("--thinfp", type=int, default=8, help='Thin the focal plane by how much?')
    parser.add_argument("--group-size", type=int, default=1, help='Group size')
    parser.add_argument("--output-dir", type=str, help='Output directory')
    parser.add_argument("--noise", default=False, action='store_true', help='Set to add noise, default=False')

    o = parser.parse_args()
    man = PipelineManager(o.globals)

    sorter = getattr(man, o.sim_sorter)
    file_input_list = sorter(o.first_sim, o.num_sims, o.output_dir, which='input')
    file_output_list = sorter(o.first_sim, o.num_sims, o.output_dir, which='filtered')
    
    for fin, fout in zip(file_input_list, file_output_list):
        outdir = fout.replace('.fits','/')
        # Filter
        maps, weighted_maps, weights = filter_map(man, fin, outdir, sched_path=o.schedule, thinfp=o.thinfp, instrument=o.telescope,
                                                 band=o.bands, sample_rate=o.sample-rate, add_noise=o.noise)
        # Write to file 
        hp.write_map(fout, maps, overwrite=True)

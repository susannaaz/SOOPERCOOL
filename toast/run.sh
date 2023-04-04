#!/bin/bash

export OMP_NUM_THREADS=1
sotodlibdir=~/sotodlib/workflows

logfile=toast.log

if [[ -e $logfile ]]; then
    echo "$logfile exists"
fi

outdir=toast_output
mkdir -p $outdir

echo "Writing $logfile"

TOAST_LOGLEVEL=debug mpirun -np 4 python \
$sotodlibdir/toast_so_sim.py \
 --config 2022.toml \
 --schedule schedules/schedule_sat.fixed_el.txt \
 --bands SAT_f150 \
 --telescope SAT1 \
 --thinfp 64 \
 --sample_rate 40 \
 --sim_noise.enable \
 --sim_atmosphere_coarse.disable \
 --sim_atmosphere.disable \
 --out $outdir \
 --job_group_size 1 \
 >& $logfile
exit

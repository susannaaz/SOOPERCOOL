#/bin/bash

#if [[ ! -e patches_sat.txt ]]; then
python3 make_sat_tiles.py
#fi

# Change to TOAST3 installation directory
#toastdir=/global/homes/k/kwolz/.local/cmbenv-20220322/lib/python3.9/site-packages/toast


export TOAST_LOGLEVEL=DEBUG

# Allowable observing elevations [deg]
outdir="schedules"
FIXED_ELEVATIONS="--elevations-deg 50,60"
FREE_ELEVATIONS="--el-min-deg 50 --el-max-deg 70"
SINGLE_ELEVATION="--elevations-deg 55"

# I-V curves are expensive

ELEVATION_CHANGE="--elevation-change-limit-deg 1 --elevation-change-time 1800"

# Run all the schedules

mkdir -p $outdir

rootname=fixed_el
toast_ground_schedule \
    @schedule_sat.par \
    @patches_sat.txt \
    $FIXED_ELEVATIONS \
    $ELEVATION_CHANGE \
    --out $outdir/schedule_sat.$rootname.txt \
    >& $outdir/schedule_sat.$rootname.log &


# Analyze the schedule
wait

python3 analyze_schedule.py schedules > analysis.txt
echo "Analysis written in analysis.txt"

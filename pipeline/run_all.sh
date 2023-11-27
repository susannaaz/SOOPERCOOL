paramfile='/pscratch/sd/s/susannaz/SOOPERCOOL/paramfiles/paramfile_susanna.yaml'

echo "Running pipeline with paramfile: ${paramfile}"

#echo "Pre-processing data..."
echo "-------------------"
python3 pipeline/pre_processer.py --globals ${paramfile} --sims --plots

#echo "Running mask stage..."
#echo "---------------------"
#python3 pipeline/mask_handler.py --globals ${paramfile} --plot

#echo "Running mock stage for data..."
#echo "------------------------------"
#python pipeline/mocker.py --globals ${paramfile} --plot 
#echo "Running mock stage for sims..."
#echo "------------------------------"
#python mocker.py --globals ${paramfile} --sims

#echo "Running mcm..."
#echo "--------------"
#python mcmer.py --globals ${paramfile}
#
#echo "Running filterer for transfer"
#echo "-----------------------------"
#python filterer.py --globals ${paramfile} --transfer
#
#echo "Running filterer for sims"
#echo "-------------------------"
#python filterer.py --globals ${paramfile} --sims
#echo "Running filterer for data"
#echo "-------------------------"
#python filterer.py --globals ${paramfile} --data
#
#echo "Running cl estimation for tf estimation"
#echo "---------------------------------------"
#python pcler.py --globals ${paramfile} --tf_est
#
#echo "Running transfer estimation"
#echo "---------------------------"
#python transfer.py --globals ${paramfile}
#
#echo "Running cl estimation for validation"
#echo "------------------------------------"
#python pcler.py --globals ${paramfile} --tf_val
#
#echo "Running pcler on data"
#echo "---------------------"
#python pcler.py --globals ${paramfile} --data --plots
#
#echo "Running pcler on sims"
#echo "---------------------"
#python pcler.py --globals ${paramfile} --sims
#
#echo "Running coadder on data"
#echo "---------------------"
#python coadder.py --globals ${paramfile} --data --plots --auto
#
#echo "Running coadder on sims"
#echo "---------------------"
#python coadder.py --globals ${paramfile} --sims
#

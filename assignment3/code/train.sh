#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=08:00:00
module load python/3.5.0
FOLDER=/home/lgpu0088/deep_learning/assignment3/code/
python3 $FOLDER/a3_vae_template.py --save_path "${FOLDER}zdim_20/" --zdim 20 >> $FOLDER/zdim_20/log.log 

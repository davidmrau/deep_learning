#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=08:00:00
python3 $HOME/deep_learning/assignment2/part3/train.py --txt_file $HOME/deep_learning/assignment2/part3/der_prozess.txt --save_path $HOME/deep_learning/assignment2/part3/models/ --summary_path $HOME/deep_learning/assignment2/part3/summaries/ --save_every 200
#>> $HOME/deep_learning/assignment2/part3/log.txt

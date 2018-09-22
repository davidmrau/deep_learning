#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=08:00:00

INPUT_LENGTHS=( "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "20" "25"  "30" "35" "40" "45" "50")

FOLDER=/home/lgpu0088/deep_learning/assignment2/part1/

for len in "${INPUT_LENGTHS[@]}"
do
	python3 /$FOLDER/train.py --input_length $len --device 'cuda:0' >> accuracies.txt
done

## This is python submit bash file

rm -rf out.txt pid.txt results_advmc/* mdlparams_advmc/*

nohup /home/jqwang/anaconda3/envs/pytorch/bin/python main_advmc.py > out.txt 2>&1 & echo $! > pid.txt
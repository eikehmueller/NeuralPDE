#source /home/km2560/firedrake/bin/activate

omp_set_num_threads(1)
mkdir -p "~/internship/output"
python train.py --path_to_output_folder=~/internship/output
#!/bin/bash
#SBATCH --job-name=ncp2048
#SBATCH --partition=astro3_long
#SBATCH --account=astro
#SBATCH --nodes=5
#SBATCH --mem=24G
#SBATCH --cpus-per-task=20
#SBATCH --output=logs/ncp2048.out
#SBATCH --begin=now

module load astro #python/anaconda3/2021.05
export SRUN_CPUS_PER_TASK=20
NODELIST=$SLURM_NODELIST

partition=astro3_long
account=astro

LX=2048
out_suffix_extra="_clp"

base_path="/lustre/astro/kpr279"
sim_base_path="ns${LX}"
ana_base_path="nematic_analysis/na${LX}/na${LX}"
suffix_list=("" "l") # "l" "vl")


# Function to expand the node range into individual nodes
expand_node_range() {
    local range=$1
    local nodes=()

    # Check if the range is in the expected format
    if [[ $range =~ \[([0-9]+)-([0-9]+)\] ]]; then
        start=${BASH_REMATCH[1]}
        end=${BASH_REMATCH[2]}

        for ((i = start; i <= end; i++)); do
            nodes+=("node$i")
        done
    else
        nodes+=("$range")
    fi

    echo "${nodes[@]}"
}

# Expand the SLURM_NODELIST range into individual nodes
EXPANDED_NODELIST=$(expand_node_range $NODELIST)

echo ""
# Loop over each path in the list
for suffix in "${suffix_list[@]}"; do

    data_folder="${base_path}/${sim_base_path}${suffix}"
    

    if [ ! -d "${data_folder}" ]; then
        echo "Data path not found: $data_folder"
        continue
    fi
    echo ""
    echo "------------------------------------------------------------------------------------"
    echo "Analyzing Data path: $data_folder"
  
    defect_list_folder="${base_path}/${ana_base_path}${suffix}_cl"
    main_output_folder="${base_path}/${ana_base_path}${suffix}${out_suffix_extra}"
    echo "Output path: $main_output_folder"
    echo ""

    # set zeta values. Format: zeta_values=("0.021" "0.022" "0.023")
    zeta_values=()

    # if no zeta provided, run through all zeta values in the data folder
    if [ ${#zeta_values[@]} -eq 0 ]; then
        # Loop through subfolders in the given folder path
        for dir in "$data_folder"/*; do
            # Check if the entry is a directory
            if [ -d "$dir" ]; then
                # Extract the number from the directory name
                number=$(basename "$dir" | grep -oP '(?<=_zeta_)[0-9.]+')
        
                # Check if the extracted number is not empty
                if [ -n "$number" ]; then
                    # Add the number to the zeta_values array
                    zeta_values+=("$number")
                fi
            fi
        done
    fi

        # Create main output folder if it doesn't exist
    mkdir -p "$main_output_folder"

    for zeta in "${zeta_values[@]}"; do
        zeta_output_folder="${main_output_folder}/analysis_zeta_${zeta}"  # Define zeta-specific output folder
    
        mkdir -p "$zeta_output_folder"  # Create zeta-specific output folder if it doesn't exist

        # Initialize the job counter
        if [ "$LX" -eq 256 ] && [ "$suffix" = "vl" ]; then
            job_count=20
        else
            job_count=0
        fi

        # Submit jobs in parallel
        for node in $EXPANDED_NODELIST; do
            output_folder="${zeta_output_folder}/zeta_${zeta}_counter_${job_count}"  # Define output folder for each run
            input_folder="${data_folder}/output_test_zeta_${zeta}/output_test_zeta_${zeta}_counter_${job_count}" 
            defect_folder="${defect_list_folder}/analysis_zeta_${zeta}/zeta_${zeta}_counter_${job_count}"
            mkdir -p "$output_folder"  # Create output folder for each run if it doesn't exist
        
            # Submit jobs in parallel using srun with proper resource allocation
            srun -N 1 -n 1 -p $partition -A $account --nodelist=$node python3 python_scripts/cluster_perc.py --input_folder "$input_folder" --output_folder "$output_folder" --defect_list_folder "$defect_folder" &

            ((job_count++))  # Increment job counter

        done
        # Wait for all jobs to finish before moving on to the next zeta value
        wait 
    done
    
done
echo ""
echo "------------------------------------------------------------------------------------"
echo "All jobs completed."
echo ""



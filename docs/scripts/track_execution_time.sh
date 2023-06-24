#!/usr/bin/env bash

##########################################################################
# Script Name: Notebook Execution Time Tracker
#
# Description: This script recursively traverses the 'examples' and
#              'tutorials' directories, executing all Jupyter notebooks
#              (.ipynb files) found therein.
#
#              For each notebook, the script records its execution time
#              and stores this data in a CSV file for easy analysis.
#
#              Additionally, the script generates and stores information
#              about the Python environment where the notebooks are
#              executed. This includes the output of 'pip list',
#              displaying all installed packages, and specific information
#              for 'lambeq' and 'discopy' packages using 'pip show'.
#
#              Lastly, the script checks whether each notebook's execution
#              succeeded, storing the executed notebooks even if they
#              failed. The presence of any execution errors is noted and
#              stored as well.
##########################################################################


# Default excluded notebook
exclude_notebook=""

# Parse command line arguments
while getopts ":e:n:" opt; do
  case ${opt} in
    e)
      exclude_notebook=$OPTARG
      ;;
    n)
      n_runs=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" 1>&2
      ;;
    :)
      echo "Option -$OPTARG requires an argument" 1>&2
      ;;
  esac
done
shift $((OPTIND -1))

# if n_runs is not set, set it to 1
: ${n_runs:=1}

# Function to convert 'real' time to seconds
function convert_time() {
    # Extract minutes and seconds
    local time=$1
    local minutes=${time%m*}
    local seconds=${time#*m}; seconds=${seconds%s}

    # Convert to seconds
    local total_seconds=$(echo "$minutes * 60 + $seconds" | bc)

    echo $total_seconds
}

# Check if in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "In Virtualenv, installing some packages"
    pip install pytest nbconvert nbformat ipython ipykernel
else
    echo "Not in Virtualenv"
    # break if not in virtualenv
    exit 1
fi

# Create 'notebook_runtimes' directory if it doesn't exist
base_path="./notebook_runtimes"
if [ ! -d "$base_path" ]; then
  mkdir -p "$base_path"
fi

# Store the output of 'pip show'
lambeq_info=$(pip show lambeq)
discopy_info=$(pip show discopy)

# Extract the version numbers
lambeq_version=$(echo "$lambeq_info" | grep "^Version" | cut -d ' ' -f 2)
discopy_version=$(echo "$discopy_info" | grep "^Version" | cut -d ' ' -f 2)

# Create a timestamp for output directory
timestamp=$(date +%Y%m%d_%H%M%S)

# Add lambeq and discopy versions to the directory name
timestamp="${timestamp}_lambeq-${lambeq_version}_discopy-${discopy_version}"

# Create a new directory with timestamp
mkdir -p "$base_path/$timestamp"

# Create new directories for this run
mkdir -p "$base_path/$timestamp/notebooks"

# Store the output of 'pip list'
pip list > $base_path/$timestamp/pip_list.txt

# Store the output of 'pip show'
echo "$lambeq_info" > $base_path/$timestamp/lambeq_info.txt
echo "$discopy_info" > $base_path/$timestamp/discopy_info.txt

# The directories containing your notebooks
dirs=("examples" "tutorials")

# The output CSV file
output_file="$base_path/$timestamp/notebook_execution_times.csv"

header="notebook,"
nans=""
# Print the header of the CSV file
for ((i=0; i<$n_runs; i++)); do
    header="${header}run_$i,"
    nans="${nans}NA,"
done
header="${header}status"
echo $header > $output_file

echo "Start processing notebooks..."

# Iterate over the directories
for dir in "${dirs[@]}"; do
    echo "Processing directory: $dir"

    # Iterate over the .ipynb files in each directory
    find $dir -name "*.ipynb" | while read notebook; do
        echo "Executing notebook: $notebook"
        # Extract file name
        file_name=$(basename $notebook)

        # Skip notebooks if flag was set
        if [ "$file_name" == "$exclude_notebook" ]; then
            echo "Skipping notebook: $notebook"
            echo "$notebook,${nans}skipped" >> $output_file
            continue
        fi

        status="success"
        times=()
        # Run the notebook n times
        for ((i=0; i<$n_runs; i++)); do
            # Execute the notebook and capture the output and time
            output=$(
                (time jupyter nbconvert --execute --allow-errors --to notebook \
                --output-dir "$base_path/$timestamp/notebooks" "$notebook") 2>&1
            )
            python scripts/check_errors.py \
                "$base_path/$timestamp/notebooks/$file_name"
            retval=$?

            # Extract the execution time from the output
            exec_time=$(echo "$output" | grep "real" | awk '{print $2}')
            exec_time=$(convert_time $exec_time)
            times+=($exec_time)

            # Check if nbconvert succeeded
            if [ $retval -ne 0 ]; then
                echo "Notebook execution failed with return code $retval."
                status="failed"
            fi
        done

        # generate output string
        output_string=""
        sum=0
        for time in "${times[@]}"; do
            output_string="${output_string}${time},"
            sum=$(echo "$sum + $time" | bc)
        done
        mean=$(echo "scale=2; $sum / $n_runs" | bc)

        # Write the notebook path and execution time to the CSV file
        echo "$notebook,$output_string$status" >> $output_file
        echo "Mean execution time: $mean seconds (status: $status)"
    done
done

echo "Completed processing all notebooks."

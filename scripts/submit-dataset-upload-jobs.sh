#! /bin/bash

get_abs_filename() {
	# Get the absolute path to the slurm script
	# $0 : absolute path to this file
	# $1 : path to script within `slurm/`` subdir
	echo "$(cd "$(dirname "$0")" && pwd)/$1"
}

submit_jobs() {
	echo "Submitting jobs for dataset variant: $DATASET_VARIANT"
	# Upload the raw HF datasets (after we are done preprocessing instances)
	upload_raw_dataset_job_id=$(sbatch --export=DATASET_VARIANT=$DATASET_VARIANT --parsable "$(get_abs_filename slurm/upload-raw-dataset.sh)")
	echo "Upload raw dataset: $upload_raw_dataset_job_id"

	# Upload the preprocessed HF datasets
	upload_preprocessed_dataset_job_id=$(sbatch --export=DATASET_VARIANT=$DATASET_VARIANT --parsable "$(get_abs_filename slurm/upload-preprocessed-dataset.sh)")
	echo "Upload preprocessed dataset: $upload_preprocessed_dataset_job_id"
}

DATASET_VARIANT='original'
submit_jobs
DATASET_VARIANT="reworded"
submit_jobs

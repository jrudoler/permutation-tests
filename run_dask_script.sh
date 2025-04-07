#!/bin/bash
#$ -N run_dask_sims
#$ -o $JOB_NAME-$JOB_ID.log
#$ -j y
#$ -l m_mem_free=20G
#$ -l h_rt=24:00:00
#$ -cwd 

poetry run python run_cv_simulations.py
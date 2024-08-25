#!/bin/bash
#$ -N aggregate_results
#$ -o $JOB_NAME-$JOB_ID.log
#$ -j y
#$ -l m_mem_free=60G
#$ -l h_rt=24:00:00
#$ -cwd 

poetry run python aggregate_results.py
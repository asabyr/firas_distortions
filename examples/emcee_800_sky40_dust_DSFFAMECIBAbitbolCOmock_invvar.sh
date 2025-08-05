#!/bin/sh
#SBATCH --account=hill
#SBATCH --job-name=firas_mcmc_800_sky40_dust_DSFFAMECIBAbitbolCOmock_invvar
#SBATCH -c 1
#SBATCH --mem-per-cpu=50gb
#SBATCH --time=00-12:00
#SBATCH --output=slurm_output/firas_mcmc_800_sky40_dust_DSFFAMECIBAbitbolCOmock_invvar-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=as6131@columbia.edu

cd /moto/home/as6131/firas_distortions/code/

module load anaconda/3-2022.05
conda init bash
conda activate firas
python run_mcmc.py invvar/dust_sync_CIB_CO_FF_AME/800_sky40_dust_DSFFAMECIBAbitbolCOmock_invvar.ini

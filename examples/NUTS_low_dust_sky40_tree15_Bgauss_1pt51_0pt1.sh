#!/bin/sh
#SBATCH --account=hill
#SBATCH --job-name=firas_mcmc_low_dust_sky40_tree15_Bgauss_1pt51_0pt1
#SBATCH -c 1
#SBATCH --mem-per-cpu=50gb
#SBATCH --time=00-03:00
#SBATCH --output=slurm_output/firas_mcmc_low_dust_sky40_tree15_Bgauss_1pt51_0pt1-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=as6131@columbia.edu

cd /moto/home/as6131/firas_distortions/code/

module load anaconda/3-2022.05
source activate /moto/home/as6131/.conda/envs/numpyro
/moto/home/as6131/.conda/envs/numpyro/bin/python run_NUTs.py dust/low_dust_sky40_tree15_Bgauss_1pt51_0pt1.ini

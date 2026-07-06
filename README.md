# MasterThesis_scalableGP_Modi
Code repository for Master Thesis "Scalable genetic programming to evolve high-dimensional control policies." The repository contains notebooks, code and results to reproduce all experimentation. 

The following experiments were conducted:
Kozax:
Kozax experimentation was done using https://github.com/sdevries0/Kozax/tree/main. This Kozax implementation was tested on both Brax and Mujoco

Modi (+ multi-tree)
A Modi adaptation of Kozax was implemented using https://github.com/sdevries0/Kozax/tree/MODI. However, local experimentation and changes have been implemented, so the used code is presented inside this repository. Multi-tree Modi uses the same code, but with some parameter changes. Both Modi implementation were tested on both Brax and Mujoco. 

CGPAX + RL
CPGAX and the RL methods used in this Thesis (PPO and SAC) have been implemented using https://github.com/giorgia-nadizar/cgpax/tree/main in Brax. However, an updated version is added to this repository. For MuJoCo, the MuJoCo playground notebook (https://github.com/google-deepmind/mujoco_playground/tree/main) have been used and adapted to obtain results using the RL methods.

The following packages were used to run Kozax and Modi on GPU:

Brax installation:
!pip install -U "jax[cuda12]"
!pip install brax
!pip install gymnax
!pip install jaxtyping
!pip install sympy

Mujoco installation:
!pip install -U "jax[cuda12]"
!pip install mujoco
!pip install mujoco_mjx
!pip install brax
!pip install playground
!pip install gymnax
!pip install jaxtyping
!pip install playground warp-lang==1.12.1
!pip install sympy

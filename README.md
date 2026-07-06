# MasterThesis_scalableGP_Modi
Code repository for Master Thesis "Scalable genetic programming to evolve high-dimensional control policies." The repository contains notebooks, code and results to reproduce all experimentation. 

The following experiments were conducted:
Kozax:
Kozax experimentation was done using https://github.com/sdevries0/Kozax/tree/main. This Kozax implementation was tested on both Brax and Mujoco

Modi (+ multi-tree)
A Modi adaptation of Kozax was implemented using https://github.com/sdevries0/Kozax/tree/MODI. However, local experimentation and changes have been implemented, so the used code is presented inside this repository. Multi-tree Modi uses the same code, but with some parameter changes. Both Modi implementation were tested on both Brax and Mujoco. 

CGPAX + RL
CPGAX and the RL methods used in this Thesis (PPO and SAC) have been implemented using https://github.com/giorgia-nadizar/cgpax/tree/main in Brax. For MuJoCo, the MuJoCo playground notebook (https://github.com/google-deepmind/mujoco_playground/tree/main) have been used and adapted to obtain results using the RL methods.

For more specific details, see the README files in the corresponding folder. 

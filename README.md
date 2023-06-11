# `IMRPhenomXODE`

`IMRPhenomXODE` is a new phenomenological frequency-domain waveform approximant for gravitational-wave (GW) signals from precessing binary black holes (BBHs) with generic spin configurations. 
We build upon the success of [`IMRPhenomXPHM`](https://ui.adsabs.harvard.edu/abs/2021PhRvD.103j4056P/abstract), which is one of the most widely adopted waveform approximants in GW data analyses that include spin precession, and introduce two additional significant improvements. 
First, we employ an efficient technique to numerically solve the (next-to)^4-leading-order post-Newtonian precession equations, allowing us to accurately determine the evolution of the Euler angles that describe the orientation of the orbital angular momentum. 
Secondly, we recalibrate the phase of GW modes in the frame co-precessing with the orbital angular momentum against [`SEOBNRv4PHM`](https://ui.adsabs.harvard.edu/abs/2020PhRvD.102d4055O/abstract) to capture effects arising from, e.g., variations in the spin component aligned with the orbital angular momentum due to precession. 

The code is described in https://arxiv.org/abs/xxxx.xxxxx, which can also be accessed via the LIGO DCC: https://dcc.ligo.org/LIGO-P2300104. 

## Installation
```bash
git clone git@github.com:hangyu45/IMRphenomXODE.git
conda create --name <environment_name> --file requirements.txt --channel conda-forge
```
(replace `<environment_name>` by a name of your choice).

## Quick Start Guide

Please see [waveform_demo.ipynb](waveform_demo.ipynb) for an example.

## Contact

Hang Yu

Email: hang.yu2@montana.edu

# `IMRPhenomXODE`

We present IMRPhenomXODE, a new phenomenological frequency-domain waveform approximant for gravitational-wave (GW) signals from precessing binary black holes (BBHs) with generic spin configurations. We build upon the success of IMRPhenomXPHM [G. Pratten et al., Phys. Rev. D 103, 104056 (2021), which is one of the most widely adopted waveform approximants in GW data analyses that include spin precession, and introduce two additional significant improvements. First, we employ an efficient technique to numerically solve the (next-to)$^4$-leading-order post-Newtonian precession equations, which allows us to accurately determine the evolution of the orientation of the orbital angular momentum $\boldsymbol{\hat{L}}{\rm N}$
even in cases with complicated precession dynamics, such as transitional precession. Second, we recalibrate the phase of GW modes in the frame coprecessing with 
$\boldsymbol{\hat{L}}{\rm N}$
against SEOBNRv4PHM [S. Ossokine et al., Phys. Rev. D 102, 044055 (2020)] to capture effects due to precession such as variations in the spin components aligned with $\boldsymbol{\hat{L}}{\rm N}$. 
By incorporating these new features, IMRPhenomXODE achieves matches with SEOBNRv4PHM that are better than 99% for most BBHs with mass ratios $q \geq 1/6$ and with arbitrary spin configurations. In contrast, the mismatch between IMRPhenomXPHM and SEOBNRv4PHM often exceeds 10% for a BBH with $q\lesssim 1/2$ and large in-plane or antialigned spin components. Our implementation is also computationally efficient, with waveform evaluation times that can even be shorter than those of IMRPhenomXPHM for BBH signals with long durations and hence high frequency resolutions. The accuracy and efficiency of IMRPhenomXODE position it as a valuable tool for GW event searches, parameter estimation analyses, and the inference of underlying population properties.

The code is described in https://arxiv.org/abs/2306.08774, which can also be accessed via the LIGO DCC: https://dcc.ligo.org/LIGO-P2300104. 

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

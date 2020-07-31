# VWAVES-GaiaDR2

## Overview

This repository includes all the code required to reproduce the results in Bennett & Bovy (2020, in prep.) which models the interaction between a passing satellite and the Galactic disc. We use this model to test whether or not Sgr could have caused the observed perturbation in the vertical disc.

## AUTHOR
Morgan Bennett - bennett at astro dot utoronto dot ca

## Code

### 1. [SgrModels.ipynb](SgrModels.ipynb)

Looks at different mass models of the Sgr dwarf galaxy ranging between $5*10^{10}$ M$_\odot$ and $5*10^{8}$ M$_\odot$. The Sgr mass models consist of two Hernquist components, stellar and dark matter. We also calculate the half-mass radius to use in our dynamical friction calculations. 

### 2. [AnalyzeSgr.ipynb](AnalyzeSgr.ipynb)

Reads in the orbits of Sgr in the five different Milky Way potenials and with the five different Sgr mass models. Calculates important properties of the orbits such as the time since passing through the disc and how much time has passed since Sgr passed through the disc. 

### 3. [BuildingAModel.ipynb](BuildingAModel.ipynb)

Explores a simple model of the interaction between the verical disc and the Milky Way disc. Considers the impacts of changing the properties of the disc as well as some considerations of the model.

### 4. [CompareSim2Model.ipynb](CompareSim2Model.ipynb)

Compares the 1-dimensional model to the 1-dimensional n-body simulation. This allows us to test how well the linear perturbation is able to capture the dynamics of the non-linear behaviour.

### 5. [RealisticModel.ipynb](RealisticModel.ipynb)

We use the modeling machinery developed above toanswer the question of whether the perturbation to the local solarneighbourhood due to the passage of the Sgr dwarf galaxy can bethe cause of the observed asymmetry in the vertical number counts, the bending-like perturbation to the mean vertical velocity, and the Gaia phase-space spiral.

## Publications

# WattCast - *Electricity Load Forecasting From County to Household*


Copyright (c) 2023 Nikolaus Houben
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wattcast/SFH%20Load%20Forecasting?workspace=user-nikolaushouben](https://wandb.ai/wattcast/WattCast)

This repository applies the following methods:

> * Box-Cox Transform to rectify the distrubtion of the data 
> * Cyclic and relative datetime encodings 
> * Deep Learning forecasting method, such as the Temporal Fusion Transformer
> * All deep learning models are publically available @ [wandb model training](https://wandb.ai/nikolaushouben/load_forecasting_lbl)

Furthermore, new data has been introduced to discuss the differnces in forecasting approaches for electricity load on various spatial and temporal scales.

> * Disaggregated Single-Family Home Electricity Load Data [Schlemminger et al. 2022](https://zenodo.org/record/5642902#.ZBjEVcLMIuU)
> * Aggregated Neighborhood-level

The forecasts in this repo were generated with the [darts](https://unit8co.github.io/darts/README.html), [pytorch](https://pytorch.org/), and dmlc [XGBoost](https://xgboost.ai/) and experiment tracking was performed with [weights & biases](https://wandb.ai/site).


Further readings:

> * Abstract submitted to E-Energy 2023 – SoDec Workshop: [Link]()
> * Tutorial-Style Notebook for Day-Ahead Household Load Forecasting: [Link](https://wandb.ai/wattcast/XGBoost/reports/A-Recipe-to-Forecast-the-Electricity-Load-in-a-Household-without-Deep-Learning--Vmlldzo0MjAwNzAz)

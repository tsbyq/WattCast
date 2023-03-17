# WattCast - *Electricity Load Forecasting From County to Household*

This is a forked repository from following paper:

> *Zhe Wang, Han Li, Tianzhen Hong, Mary Ann Piette. 2021. Predicting City-Scale Daily Electricity Consumption Using Data-Driven Models. Submitted to Advance in Applied Energy*

<!--
[[slides]](docs/slides.pdf)[[paper]](https://dl.acm.org/doi/10.1145/3408308.3427980)
-->

## Extensions

Copyright (c) 2022 Nikolaus Houben

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository applies novel methods to forecast the original county level load data. Specifically:

> * Box-Cox Transform to rectify the distrubtion of the data
> * Cyclic and relative datetime encodings 
> * Deep Learning forecasting method, such as the Temporal Fusion Transformer
> * All deep learning models is publically available [wandb model training](https://wandb.ai/nikolaushouben/load_forecasting_lbl)

Furthermore, new data has been introduced to discuss the differnces in forecasting approaches for electricity load on various spatial and temporal scales.

> * Disaggregated Single-Family Home Electricity Load Data [zenedo data source](https://www.example.com)
> * Aggregated Neighborhood-level 


The framework used in this repository is the darts forecasting library.

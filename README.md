# WattCast: Electricity-Load-Forecasting-From-County-to-Household

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

Furthermore new data has been introduced: TBC


The framework used in this repository is the darts forecasting library.

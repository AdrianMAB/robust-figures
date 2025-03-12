# Robustness of Distributed Vulnerability using a random synthetic precipitation generator

This repository contains the figures of the article. The methodology combines the TETIS model, a series of composite indexes and a probabilistic precipitation generator. The content is:

* The observed and simulated hydrographs.
* Discrete probability distribution function.
![Average Rain probability](generator/discrete/average.svg)
![Rain probability for all stations](generator/discrete/full.svg)
* Continuous probability distribution function.
* Similarity between synthetic data and observed precipitation.
* Ranges of variation of distributed vulnerability.
![Variation of global vulnerability over one cell for all the iterations](vulnerability/high_vul_boxplot.svg)

### Animations
This folder have animated GIFs of the lowest, average and highest values of vulnerabilities of iterations. [**View more.**](vulnerability/gif/)

| **Minimum**            |  **Average**               |   **Maximum**          |
:-------------------------:|:-------------------------:|:-------------------------:
![Global vulnerability (lower iteration)](vulnerability/gif/IVG_vulmin.gif) | ![Global vulnerability (average iteration)](vulnerability/gif/IVG_vulaverage.gif) | ![Global vulnerability (higher iteration)](vulnerability/gif/IVG_vulmax.gif)

| **Minimum**            |  **Average**               |
:-------------------------:|:-------------------------:
![Global vulnerability (lower iteration)](vulnerability/gif/IVG_vulmin.gif) | ![Global vulnerability (average iteration)](vulnerability/gif/IVG_vulaverage.gif)
| **Maximum**            |                                       |
:-------------------------:|:-------------------------:
![Global vulnerability (higher iteration)](vulnerability/gif/IVG_vulmax.gif)
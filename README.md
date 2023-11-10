# SANN (Subtree-based Attention Neural Network)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the code implementation of the SANN (Subtree-based Attention Neural Network) model, as presented in the paper titled "SANN: Programming Code Representation Using Attention Neural Network with Optimized Subtree Extraction." You can access the paper [here](https://doi.org/10.1145/3583780.3615047).

### Disclaimer
Please note that the current organization of the code is not final, and the current version has some missing parts (due to unpublished works) as this is part of an ongoing project. The repository will be updated with a more organized and more structured version in the near future. Feel free to modify the content and structure as needed for your specific project.

## Code Structure

The code folder includes two main files:

- **parserBFS.py**: This file contains the code to parse Java programs using the javalang parser and formats the programs for the SANN model's use.

- **model.py**: This file comprises the preprocessing and model training code. The current version of the code is designed for a specific task (program correctness prediction). To achieve optimal results, it is recommended to tune the model hyperparameters according to the task it is being trained on. Additional information on hyperparameter tuning can be found in the paper.

## Usage

To utilize the SANN model for your specific task, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/SANN.git

2. Navigate to the repository:

   ```bash
   cd SANN

3. Install the necessary dependencies. You may use a virtual environment for this (will be uploaded soon):

   ```bash
   pip install -r requirements.txt

4. Customize the model parameters and configurations in the code files if needed.

5. Execute the code to train or use the SANN model.

6. Add testing code in this version.

### Citation

If you find this code or the SANN model useful for your work, please cite the following paper:
```bash
Hoq, M., Chilla, S. R., Ahmadi Ranjbar, M., Brusilovsky, P., & Akram, B. (2023, October). SANN: Programming Code Representation Using Attention Neural Network with Optimized Subtree Extraction. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (pp. 783-792).



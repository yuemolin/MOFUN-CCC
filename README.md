# MOFUN-CCC
**M**ulti **O**mics **FU**sion neural **n**etwork - **C**omputational **C**ell **C**ounting

MOFUN-CCC is a multi-modal deep learning algorithm that operates under a supervised framework, leveraging intermediate fusion techniques to process bulk gene expression and bulk DNA methylation data. Its primary objective is to generate absolute cell counts as its output.

TODO: 
1. Add program image
2. Add widget
3. Add tutorials for train and predict
4. add trained model
5. test on machine without gpu
6. add GUI


## Table of Contents:

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Arguments](#arguments)
- [Examples](#examples)

## Introduction

Provide an introduction to your project. Explain what it does and why someone might want to use it.

## Installation

```bash
# Create Conda Environment
conda create -n MOFUN_CCC python=3.10.8 -y
conda activate MOFUN_CCC

# Install pytorch for GPU
pip3 install torch torchvision torchaudio

# Change into the project directory
git clone https://github.com/yuemolin/MOFUN-CCC.git
cd MOFUN-CCC
pip install -r requirements.txt
```

## Usage
Explain how to use your script or project. You can include simple usage examples here.
```
python MOFUN_CCC/Main_train.py --Count <counts.txt> --RNA <rna.txt> --DNAm <dnacpg.txt> --Output <output_folder>
```
## Examples
Provide some usage examples to demonstrate how your script can be used in different scenarios.
# Example 1: Basic usage
python your_script.py --Count counts.txt --RNA rna.txt --DNAm dnacpg.txt --Output output_folder

# Example 2: Specify marker file and method
python your_script.py --Count counts.txt --RNA rna.txt --DNAm dnacpg.txt --Output output_folder --GEP_Marker markers.txt --Marker_Method FC

# Example 3: Change model parameters
python your_script.py --Count counts.txt --RNA rna.txt --DNAm dnacpg.txt --Output output_folder --Model custom_model.json --Loss L2loss --Activation Celu


## Arguments
List and describe each command-line argument your script accepts. Include the argument name, type, whether it's required or has a default value, and a brief description of its purpose.

--Count (required, str): Tab-separated TXT cell counts file (rows: Samples, columns: Cell types).
--RNA (required, str): Tab-separated TXT gene TPM file (rows: Genes, columns: Samples).
--DNAm (required, str): Tab-separated TXT CpG Beta file (rows: CpGs, columns: Samples).
--Output (required, str): Output folder path for model saving.
--GEP_Marker (str): Path of the marker file, list of Gene names or Association matrix, generate when leaving blank.
--DNAm_Marker (str): Path of the marker file, list of CpG names or Association matrix, generate when leaving blank.
--Marker_Method (str, choices=["FC", "P"]): Marker selection method, FC: by fold change, P: by p-value.
--RNA_Marker_num (int, default=6000): Number of RNA markers to select based on Marker_Method.
--DNAm_Marker_num (int, default=6000): Number of DNAm markers to select based on Marker_Method.

## Contributing

If you find a bug :bug:, please open a [bug report](https://github.com/yuemolin/MOFUN-CCC/issues).
If you have an idea for an improvement or new feature :rocket:, please open a [feature request](https://github.com/yuemolin/MOFUN-CCC/issues).

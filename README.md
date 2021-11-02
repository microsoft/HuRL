# Heuristic Guided Reinforcement Learning (HuRL)


This repository contains codes to reproduce the experimental results of the Heuristic Guided Reinforcement Learning paper published in NeurIPS 2021 by Ching-An Cheng, Andrey Kolobov, and Adith Swaminathan.

## Installation

Create a conda environment with python 3.7 and install the repository.

    conda create -n hurl pip python=3.7
    conda activate hurl
    git clone https://github.com/microsoft/HuRL
    cd HuRL
    pip install -e .

The codes use Mujoco environments, visit https://www.roboti.us/license.html for Mujoco license key.

## Execution

To run the code, simply run `python main.py` in the repository directory.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

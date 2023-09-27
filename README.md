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


## In this README :

- [Features](#features)
- [Usage](#usage)
  - [Initial setup](#initial-setup)
  - [Creating releases](#creating-releases)
- [Contributing](#contributing)

## Usage

### Initial setup

1. [Create a new repository](https://github.com/allenai/python-package-template/generate) from this template with the desired name of your project.

    *Your project name (i.e. the name of the repository) and the name of the corresponding Python package don't necessarily need to match, but you might want to check on [PyPI](https://pypi.org/) first to see if the package name you want is already taken.*

2. Create a Python 3.8 or newer virtual environment.

    *If you're not sure how to create a suitable Python environment, the easiest way is using [Miniconda](https://docs.conda.io/en/latest/miniconda.html). On a Mac, for example, you can install Miniconda using [Homebrew](https://brew.sh/):*

    ```
    brew install miniconda
    ```

    *Then you can create and activate a new Python environment by running:*

    ```
    conda create -n my-package python=3.9
    conda activate my-package
    ```

3. Now that you have a suitable Python environment, you're ready to personalize this repository. Just run:

    ```
    pip install -r setup-requirements.txt
    python scripts/personalize.py
    ```

    And then follow the prompts.

    :pencil: *NOTE: This script will overwrite the README in your repository.*

4. Commit and push your changes, then make sure all GitHub Actions jobs pass.

5. (Optional) If you plan on publishing your package to PyPI, add repository secrets for `PYPI_USERNAME` and `PYPI_PASSWORD`. To add these, go to "Settings" > "Secrets" > "Actions", and then click "New repository secret".

    *If you don't have PyPI account yet, you can [create one for free](https://pypi.org/account/register/).*

6. (Optional) If you want to deploy your API docs to [readthedocs.org](https://readthedocs.org), go to the [readthedocs dashboard](https://readthedocs.org/dashboard/import/?) and import your new project.

    Then click on the "Admin" button, navigate to "Automation Rules" in the sidebar, click "Add Rule", and then enter the following fields:

    - **Description:** Publish new versions from tags
    - **Match:** Custom Match
    - **Custom match:** v[vV]
    - **Version:** Tag
    - **Action:** Activate version

    Then hit "Save".

    *After your first release, the docs will automatically be published to [your-project-name.readthedocs.io](https://your-project-name.readthedocs.io/).*

## Contributing

If you find a bug :bug:, please open a [bug report](https://github.com/yuemolin/MOFUN-CCC/issues).
If you have an idea for an improvement or new feature :rocket:, please open a [feature request](https://github.com/yuemolin/MOFUN-CCC/issues).

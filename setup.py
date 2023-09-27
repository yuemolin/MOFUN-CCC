from setuptools import setup, find_packages
setup(
    name = 'MOFUN-CCC',
    version = '1.1.0',
    description = 'Multi Omics FUsion Neural network- Computational Cell Counting',
    author = 'Molin Yue',
    author_email = 'moy6@pitt.edu',
    url = 'https://github.com/yuemolin/MOFUN-CCC',
    license = 'MIT License',
    packages = find_packages(),
    python_requires='>=3.9',
    platforms = 'any',
    install_requires = [
        'matplotlib',
        'numpy>=1.24.2',
        'pandas>=1.5.2',
        'scikit-learn>=1.0.2',
        'scipy>=1.8.1',
        'statsmodels>=0.13.5',
        'torch>=1.13.0',
        'torchmetrics',
        'tqdm'
    ],
)
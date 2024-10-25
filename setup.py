from setuptools import setup, find_packages

# Read the contents of the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pgnn",  # Package name
    version="0.1.0",  # Initial version
    author="Rogério A. Gouvêa",  # Add your name or the authorship group
    author_email="rogeriog.em@gmail.com",  # Add your email
    description="A package that uses pretrained graph-neural network models as featurizers for materials structures.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rogeriog/pGNN",  # URL to the project
    packages=find_packages(),  # Automatically find all packages
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License", 
    ],
    python_requires='>=3.8',  # Specify Python version compatibility
    install_requires=[
    "tensorflow>=2.0",
    "scikit-learn>=0.24.0",
    "h5py>=2.10.0",
    "pandas>=1.1.0",
    "numpy>=1.18.0",
    "megnet>=1.0.0",
    "keras>=2.3.0",
    "contextlib2",  # for managing stdout redirection
    "pickle-mixin",  # for loading and saving scalers
    "pymatgen>=2022.0.0",  # for handling crystal structures
   ],
    package_data={
         'pgnn': ['featurizers/custom_models/*.h5', 'featurizers/custom_models/*.pkl'], 
    },
    entry_points={
        'console_scripts': [
            # Add any scripts you want to run directly from the command line
        ],
    },
)

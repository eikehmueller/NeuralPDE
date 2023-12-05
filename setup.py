"""Setup script for module

To install, use

    python -m pip install .

or, for an editable install,

    python -m pip install --editable .

"""

from setuptools import setup

long_description = """
Python classes for implementing neural PDE solver on the sphere.

For further details and information on how to use this module, see README.md
"""

# Extract requirements from requirements.txt file
with open("requirements.txt", "r", encoding="utf8") as f:
    requirements = [line.strip() for line in f.readlines()]

# Run setup
setup(
    name="neural_pde",
    author="Eike Mueller",
    author_email="e.mueller@bath.ac.uk",
    description="Neural PDE solver on the sphere",
    long_description=long_description,
    version="1.0.0",
    packages=["neural_pde"],
    package_dir={"": "src"},
    install_requires=[
        'importlib-metadata; python_version == "3.8"',
    ]
    + requirements,
    url="https://github.com/eikehmueller/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0 License",
        "Operating System :: POSIX :: Linux",
    ],
)

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="enlnipypetools",
    version="0.0.1",
    author="Andrew Stier",
    author_email="andrewstier@uchicago.edu",
    description="Tools for nipype",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/enlberman/enlnipypetools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

import setuptools

with open("../README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="i3pubtools-thejevans",
    version="0.0.1",
    author="John Evans",
    author_email="john.evans@icecube.wisc.edu",
    description="IceCube analysis tools for use with public data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thejevans/umd_icecube_analysis_tutorial",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
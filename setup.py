import setuptools

with open("./README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mla-thejevans",
    version="0.0.1",
    author="John Evans, Jason Fan, Michael Larson",
    author_email="john.evans@icecube.wisc.edu",
    description="IceCube analysis tools for use with public data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thejevans/mla",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

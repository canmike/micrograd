from setuptools import setup, find_packages

setup(
    name="mikegrad",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
    ],
    author="Can Michael Hucko",
    author_email="c.michaelhucko@gmail.com",
    description="A minimal autograd framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/canmike/mikegrad",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Adjust based on your license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Adjust based on your minimum Python version
)

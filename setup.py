 
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "numpy",
    "pandas >= 0.25",
    "keras >= 2.3",
    "sklearn >= 0.21",
    "yaml",
    "bokeh",
    "tables",
    "six",
    "astropy"
]

setuptools.setup(
    name="nnaps", # Replace with your own username
    version="0.0.1",
    author="Joris Vos",
    author_email="joris.vos@uv.cl",
    description="Neural Network assisted Population Synthesis tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vosjo/nnaps",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires='>=3.7',
)

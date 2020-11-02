 
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

version = {}
with open("nnaps/version.py") as fp:
    exec(fp.read(), version)

install_requires = [
    "numpy",
    "pandas >= 0.25",
    "keras >= 2.3",
    "xgboost >= 0.9",
    "scikit-learn >= 0.22.1",
    "pyyaml",
    "matplotlib",
    "bokeh",
    "tables",
    "six",
    "astropy",
]

setuptools.setup(
    name="nnaps",
    version=version['__version__'],
    author="Joris Vos",
    author_email="joris.vos@uv.cl",
    description="Neural Network assisted Population Synthesis tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vosjo/nnaps",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    test_suite='pytest.collector',
    tests_require=['pytest'],
    entry_points = {
        'console_scripts': ['nnaps-mesa=nnaps.mesa.main:main'],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires='>=3.7',
)


Dealing with MESA models
========================

The first part of NNaPS package focuses on extracting useful information from a set of MESA models. To do this, NNaPS
contains 2 command line tools. One to convert a MESA binary run to the compressed HDF5 format, and the second to
extract interesting properties from the run.

Both tools run from the command line as follows:

.. code-block:: bash

    nnaps-mesa compress
    nnaps-mesa extract

Compressing runs (compress)
---------------------------

The 'compress' tool collects all useful information from a MESA run and stores it in a hdf5 file. This has the advantage
that a lot of disc space is saved. Especially when processing MESA runs on a laptop, this will be very useful. As a
comparison. A typical MESA binary run with a low mass primary evolving to the He core burning phase takes up roughly
22 Mb. After compression, the hdf5 file is only 1.4 Mb.

**nnaps-mesa** compress [*options*]

.. program:: nnaps-mesa compress

.. option:: -i, -inputdir (str) <input directory path>

    The directory containing all mesa models. Each model in its own sub folder.

.. option:: -f, -infofile (str) <info file path>

    Path to a csv file containing a list of all the models that you want to extra, with potentially extra information
    to add to the individual models. This file needs to contain at least 1 column with the name of the folder containing
    the MESA model. This collumn needs to be called 'path'. Example of such a file:

    .. code-block:: bash

        path,extra_info1, extra_info2
        model_dir1,10,disk
        model_dir2,20,halo

.. option:: -o, -outputdir (str) <output directory path>

    The path to the directory where you want the compressed hdf5 MESA files to be stored.

.. option:: -s, -setup (str) <setup file path>

    yaml file containing the settings the compress action.

    If not setup file is give, nnaps-mesa will look for one in the current directory or in the *<user>/.nnaps*
    directory. In that case the filename of the setup file needs to be *defaults_compress.yaml*.

    If no setup file can be found anywhere, nnaps-mesa will use the defaults stored in the mesa.defaults module.

.. option:: --skip

    When provided, nnaps-mesa will only compress models that are not yet present in the output folder. Models that
    already have a compressed hdf5 version in the output folder will be ignored.


Basic usage
^^^^^^^^^^^

The most simple way to use the compress tool is to provide the folder where all MESA models are located, and the folder
where you want the compressed files to be stored:

.. code-block:: bash

    nnaps-mesa compress -i <input folder> -o <output folder>

compress will use standard settings assuming the following file structure for a MESA run:

::

    MESA model
    ├── LOGS
    │   ├── binary_history.data
    │   ├── history1.data
    │   ├── history2.data
    │   ├── profile1.profile
    │   ├── profile2.profile
    │   └── profile3.profile
    ├── inlist_project
    ├── inlist1
    ├── inlist2
    └── log.txt

The binary and stellar history files are located in the LOGS directory together with any potential profiles. The
terminal output of the MESA run is stored in the log.txt file. By default the binary and stellar history will be
compressed together with all profiles found. Compress will also extract the stopping condition from the terminal output
if possible. The compressed hdf5 file has the following structure.

::

    MESA model
    ├── extra_info
    │   └── termination_code
    ├── history
    │   ├── binary
    │   ├── star1
    │   └── star2
    ├── profile_legend
    └── profiles
        ├── profile1
        ├── profile1
        └── profile3

profile_legend is an array containing the model_number when the profile is taken together with the profile name. Both
for profiles and history files, only the actual data is saved, not the header info!

For more options on compressing models see :doc:`mesa_compress`


Extracting parameters (extract)
-------------------------------

After compressing all the MESA models, it is time to extract some interesting parameters. This is done with the
`nnaps-mesa extract` command. Extract will load the MESA model, detect the stability of the model and apply a CE
ejection is requested and then extract overall parameters of the run. It can also detect which evolution phases the
component go through during the model.

**nnaps-mesa** extract [*options*]

.. program:: nnaps-mesa extract

.. option:: -i, -input (str) <input directory or csv file>

    A directory containing the compressed stellar evolution models to extract, or a csv file containing a list of all
    models to extract and optionally individual extraction options for each model. The csv file needs to contain at
    least one column with the path to the model called 'path'

.. option:: -o, outputfile (str) <output file path>

    The path to the csv file where you want the extracted parameters to be stored.

.. option:: -s, -setup (str) <setup file path>

    yaml file containing the settings the extract action.

    If not setup file is give, nnaps-mesa will look for one in the current directory or in the *<user>/.nnaps*
    directory. In that case the filename of the setup file needs to be *defaults_extract.yaml*.

    If no setup file can be found anywhere, nnaps-mesa will use the defaults stored in the mesa.defaults module.

Basic usage
^^^^^^^^^^^

The most simple way to run the extract command is to provide it with the folder where the compressed models are located
and the filename to store the extracted parameters in:

.. code-block:: bash

    nnaps-mesa extract -i <input folder> -o <output csv filename>

Using the default settings this will for each model:

    1. check if the model is stable using the default criterion: max(lg_mstar_dot_1) < -2
    2. if the model is unstable, apply the CE formalism of Iben & Tutukov 1984
    3. check if a contact binary is formed during evolution
    4. extract the default parameters (see defaults)

The exact function will then save the default parameters for each model to a csv file.

for more options on extract see :doc:`mesa_extract`



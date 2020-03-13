
Dealing with MESA models
========================

The first part of NNaPS package focuses on extracting useful information from a set of MESA models. To do this, NNaPS
contains 2 command line tools. One to convert a MESA binary run to the compressed HDF5 format, and the second to
extract interesting properties from the run.

Both tools run from the command line as follows:

.. code-block:: bash

    nnaps-mesa -2h5
    nnaps-mesa -extract

command line options
--------------------

**nnaps-mesa** mode <*(file_list)*> <*model_directory*> [*options*]

.. program:: nnaps-mesa

.. option:: -2h5 (file_list.csv) model_directory

    The compress option, used to compress MESA models and store them in hdf5 format.

    **file_list.csv** optional argument. A list of the models in csv format. Should at least contain  a column named
    'path' with the folder name of the mesa models. The other columns can contain other information which will be stored
    in the hdf5 file of the model.

    **model_directory** mandatory argument. The directory containing all mesa models. Each model in its own sub folder.

.. option:: -extract model_directory

    The extract option, used to extract model parameters from MESA models stored in hdf5 format.

    **model_directory** mandatory argument. The directory containing all mesa models in hdf5 format.

.. option:: -setup setup_file

    yaml file containing the settings for either the :option:`-2h5` or :option:`-extract` option.

    If not setup file is give, nnaps-mesa will look for one in the current directory or in the *<user>/.nnaps*
    directory. In that case the filename of the setup file needs to be *defaults_2h5.yaml* or *defaults_extract.yaml*
    for respectively the :option:`-2h5` and :option:`-extract` option.

    If no setup file can be found anywhere, nnaps-mesa will use the defaults stored in the mesa.defaults module.

.. option:: -o output

    Where the output of either the :option:`-2h5` or :option:`-extract` option should be stored.

    In the case of :option:`-2h5`: *output* should be the path to the directory where you want the hdf5 MESA files
    to be stored.

    In the case of :option:`-extract`: *output* is the name of the csv file where nnaps-mesa will write the extracted
    parameters for all models.

.. option:: --skip

    Only relevant for the :option:`-2h5` option. When provided, nnaps-mesa will only compress models that are not yet
    present in the output folder. Models that already have a compressed hdf5 version in the output folder will be
    ignored.

Compressing runs (2h5)
----------------------

The '2h5' tool collects all useful information from a MESA run and stores it in a hdf5 file. This has the advantage that
a lot of disc space is saved. Especially when processing MESA runs on a laptop, this will be very useful. As a
comparison. A typical MESA binary run with a low mass primary evolving to the He core burning phase takes up roughly
22 Mb. After compression, the hdf5 file is only 1.4 Mb.

Basic usage
^^^^^^^^^^^

The most simple way to use the 2h5 tool is to provide the folder where all MESA models are located, and the folder
where you want the compressed files to be stored:

.. code-block:: bash

    nnaps-mesa -2h5 <input folder> -o <output folder>

2h5 will use standard settings assuming the following file structure for a MESA run:

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
compressed together with all profiles found. 2h5 will also extract the stopping condition from the terminal output if
possible. The compressed hdf5 file has the following structure.

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

For more options on 2h5 see :doc:`mesa_2h5`


Extracting parameters (extract)
-------------------------------

After compressing all the MESA models, it is time to extract some interesting parameters. This is done with the
`nnaps-mesa -extract` command. Extract will load the MESA model, detect the stability of the model and apply a CE
ejection is requested and then extract overall parameters of the run. It can also detect which evolution phases the
component go through during the model.

Basic usage
^^^^^^^^^^^

The most simple way to run the extract command is to provide it with the folder where the compressed models are located
and the filename to store the extracted parameters in:

.. code-block:: bash

    nnaps-mesa -extract <input folder> -o <output csv filename>

Using the default settings this will for each model:

    1. check if the model is stable using the default criterion: max(lg_mstar_dot_1) < -2
    2. if the model is unstable, apply the CE formalism of Iben & Tutukov 1984
    3. check if a contact binary is formed during evolution
    4. extract the default parameters (see defaults)

The exact function will then save the default parameters for each model to a csv file.

for more options on extract see :doc:`mesa_extract`




MESA 2h5
========

**nnaps-mesa** -2h5 <*(file_list.csv)*> <*model_directory*> -o <*output_directory*> [*options*]

.. program:: nnaps-mesa

.. option:: -2h5 (file_list.csv) model_directory

    The compress option, used to compress MESA models and store them in hdf5 format.

    **file_list.csv** optional argument. A list of the models in csv format. Should at least contain  a column named
    'path' with the folder name of the mesa models. The other columns can contain other information which will be stored
    in the hdf5 file of the model.

    **model_directory** mandatory argument. The directory containing all mesa models. Each model in its own sub folder.

.. option:: -o output

    The path to the directory where the compressed hdf5 MESA files need to be stored.

.. option:: -setup setup_file

    yaml file containing the detailed setup for the compression.

    If not setup file is give, nnaps-mesa will look for one in the current directory or in the *<user>/.nnaps*
    directory. In that case the filename of the setup file needs to be *defaults_2h5.yaml*.

    If no setup file can be found anywhere, nnaps-mesa will use the defaults stored in the mesa.defaults module.

.. option:: --skip

    When provided, nnaps-mesa will only compress models that are not yet
    present in the output folder. Models that already have a compressed hdf5 version in the output folder will be
    ignored.

Basic usage
-----------

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

Setup file
----------

By using a custom setup file you can specify exactly what should be included in the hdf5 archive and what the exact
structure of the MESA model directory is.

The setup file has to be structured in yaml format, and can be provided using the *-setup* option.

.. code-block:: yaml

    star_columns: []
    binary_columns: []
    profile_columns: []
    input_path_kw: 'path'
    input_path_prefix: ''
    star1_history_file: 'LOGS/history1.data'
    star2_history_file: 'LOGS/history2.data'
    binary_history_file: 'LOGS/binary_history.data'
    log_file: 'log.txt'
    add_stopping_condition: True
    profile_files: []
    profiles_path: 'LOGS'
    profile_pattern: 'profile_*.data'

.. option:: star_columns (list)

    A list of all columns in the star history file to include. When empty or not included all columns will be kept.

.. option:: binary_columns (list)

    A list of all columns in the binary history file to include. When empty or not included all columns will be kept.

.. option:: profile_columns (list)

    A list of all columns in profiles to include. When empty or not included all columns will be kept.

.. option:: input_path_kw (str)

    If nnaps-mesa :option:`-2h5` is called with a file_list.csv and a model_directory, then this keyword indicates the
    name of the column in the file_list.csv that contains the path of the directory containing the MESA model relative
    to the working directory.

.. option:: input_path_prefix (str)

    If nnaps-mesa :option:`-2h5` is called with a file_list.csv and a model_directory, then this keyword indicates the
    optional prefix to be added in front of the directory given in the file_list.csv by the :option:`input_path_kw`.
    The full path relative to the current working directory is then:

    input_path_prefix + file_list.csv[input_path_kw]

.. option:: star1_history_file (str)

    The path of the history file of the first star relative to the model directory.

.. option:: star2_history_file (str)

    The path of the history file of the second star relative to the model directory.

.. option:: binary_history_file (str)

    The path of the binary history file relative to the model directory.

.. option:: log_file (str)

    The path of the logging output of MESA relative to the model directory.

.. option:: add_stopping_condition (boolean)

    When true, the stopping criteria of MESA will be extracted from the :option:`log_file` and included in the hdf5
    file.

.. option:: profile_files (list)

    A list of which profile files to include. If empty or not included all profile files that can be identified using
    the :option:`profile_pattern` keyword will be included.

.. option:: profiles_path (str)

    The path of the directory containing the profiles relative to the model directory

.. option:: profile_pattern (str)

    The pattern of the profiles to include. Will only be used when :option:`profile_files` is empty or not included.

Reading compressed files
------------------------

NNaPS provides two methods to easily read the compressed files. The :func:`~nnaps.mesa.fileio.read_hdf5` function will read any
hdf5 formatted file and return the content as a python dictionary, while the :func:`~nnaps.mesa.fileio.read_compressed_track` reads
hdf5 files created by NNaPS and returns the result in a more directly usable way. :func:`~nnaps.mesa.fileio.read_compressed_track`
returns the combined stellar and binary history in one array, dealing automatically with inequalities in time steps in
the different history files. It also returns a dictionary with other info, and if requested a dictionary containing
the profiles.

.. automodule:: nnaps.mesa.fileio
   :members: read_hdf5, read_compressed_track


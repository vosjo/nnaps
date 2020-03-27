
MESA extract
============

**nnaps-mesa** -extract <*model_directory*> -o <*output_file.csv*> [*options*]

.. program:: nnaps-mesa

.. option:: -extract model_directory

    The extract option, used to extract model parameters from MESA models stored in hdf5 format.

    **model_directory** mandatory argument. The directory containing all mesa models in hdf5 format.

.. option:: -o output

    The name of the csv file where nnaps-mesa will write the extracted parameters for all models.

.. option:: -setup setup_file

    yaml file containing the detailed setup for the compression.

    If not setup file is give, nnaps-mesa will look for one in the current directory or in the *<user>/.nnaps*
    directory. In that case the filename of the setup file needs to be *defaults_extract.yaml*.

    If no setup file can be found anywhere, nnaps-mesa will use the defaults stored in the mesa.defaults module.

Basic usage
-----------

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

Setup file
----------

Using a setup file allows you to set different parameters for the extraction including the stability criterion,
which parameters to extract and more.

The setup file has to be structured in yaml format, and can be provided using the *-setup* option.

.. code-block:: yaml

    stability_criterion: J_div_Jdot_div_P
    stability_limit: 10
    ce_formalism: 'dewi_tauris2000'
    ce_parameters: {'a_ce': 1, 'a_th': 0.5}
    ce_profile_name: 'profile_1_jdotp10.0'
    parameters: []
    phase_flags: []
    extra_info_parameters: []

.. option:: stability_criterion (str)

    The stability criterion to use when judging is the mass loss is stable. Has to be combined with
    :option:`stability_limit` to set the value above or below which the model is unstable.

.. option:: stability_limit (float)

    The value for the stability criterion above or below which the model is considered unstable. Use
    :option:`stability_criterion` to set which criterion to use.

.. option:: ce_formalism (str)

    The name of the CE formalism to use

.. option:: ce_parameters (dict)

    Custom parameters for the CE formalism

.. option:: ce_profile_name (str)

    If a CE formalism is chosen that needs a profile, you can use this argument to give the name of the profile to use.
    If no name is specified, the profile with the model number closest to when the stability criterion is trip will be
    used.

.. option:: parameters (list)

    Which parameters to extract from the models. See `Parameters`_ for an explanation on how to structure
    parameter names.

.. option:: phase_flags (list)

    Which phase flags to extract from the models.

.. option:: extra_info_parameters (list)

    Which extra info parameters to extract from the models.


Parameters
----------

To extract useful information from a MESA model you are likely interested at parameter values at a certain moment in
evolution, or during a certain evolutionary phase. *nnaps-mesa* allows you to easily extract parameters and apply
aggregate functions on a parameter during a specified phase.

A parameter to extract consists of 3 parts divided by a double underscore '__': the name of the parameter that you are
interested in, the phase or exact point in time and potentially the function to apply. Not all three parts need to be
present. Easiest way to demonstrate this is by example:

star_1_mass__init        -> mass of the primary at the start of the run
rl_1__max                -> max of the roche lobe size of the primary star
age__HeCoreBurning__diff -> Difference in age between the start and end of the He core burning phase == duration of He core burning.
T_effective__ML__min     -> The minimum of the effective temperature during the mass loss phase

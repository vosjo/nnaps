MESA Error flags
================

The different error flags that NNaPS can return and there meaning is given in the table below. The function that is
called to check all errors is: :func:`~nnaps.mesa.evolution_errors.check_error_flags`. Some of the errors are checked
by different functions. If this is the case it is indicated in the table.

============ =================================== ============================================================
 Error code   Description                         Function
============ =================================== ============================================================
 1             Maximum model number reached       /
 2             Accretor overflow detected         /
 3             Mass loss error                    :func:`~nnaps.mesa.evolution_errors.mass_loss_error`
 4             He ignition error                  :func:`~nnaps.mesa.evolution_errors.he_ignition_error`
 5             He core burning error              :func:`~nnaps.mesa.evolution_errors.he_core_burning_error`
============ =================================== ============================================================

If you want to detect all errors, the following parameters need to be included in the history file:

- age
- lg_mstar_dot_1
- lg_wind_mdot_1
- log_LHe
- log_center_T
- log_center_Rho
- c_core_mass

.. automodule:: nnaps.mesa.evolution_errors
   :members: check_error_flags, mass_loss_error, he_ignition_error, he_core_burning_error

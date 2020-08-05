MESA Common Envelope ejection
=============================

Stability
---------

Current implemented stability criteria are:

- Mdot: lg_mstar_dot_1 > value
- delta: mass_transfer_delta > value
- J_div_Jdot_div_P: 10**log10_J_div_Jdot_div_P < value
- M_div_Mdot_div_P: 10**log10_M_div_Mdot_div_P < value
- R_div_SMA: star_1_radius / binary_separation > value

An up to date list of all stability criteria can be obtained with:

.. code-block:: python

    from nnaps.mesa.common_envelope import STABILITY_CRITERIA
    print(STABILITY_CRITERIA)

If you want to check a model against one of those stability criteria you can use the `is_stable`_ function:

.. automodule:: nnaps.mesa.common_envelope
   :members: is_stable

Common Envelope
---------------

The following formalisms are defined in NNaPS-mesa:

- iben_tutukov1984: `Iben & Tutukov 1984, ApJ, 284, 719 <https://ui.adsabs.harvard.edu/abs/1984ApJ...284..719I/abstract>`_
- webbink1984: `Webbink 1984, ApJ, 277, 355 <https://ui.adsabs.harvard.edu/abs/1984ApJ...277..355W/abstract>`_
- dewi_tauris2000: `Dewi and Tauris 2000, A&A, 360, 1043 <https://ui.adsabs.harvard.edu/abs/2000A%26A...360.1043D/abstract>`_
- demarco2011: `De Marco et al. 2011, MNRAS, 411, 2277 <https://ui.adsabs.harvard.edu/abs/2011MNRAS.411.2277D/abstract>`_

An up to date list of all recognized CE formalisms can be obtained with:

.. code-block:: python

    from nnaps.mesa.common_envelope import CE_FORMALISMS
    print(CE_FORMALISMS)


Details of the different formalisms, which parameters are required from MESA for them to work, and which parameters can
be varied for each are given below.

.. automodule:: nnaps.mesa.common_envelope
   :members: iben_tutukov1984, webbink1984, dewi_tauris2000, demarco2011

If you want to apply the CE formalism yourself on a mesa model you can use the `apply_ce`_ function:

.. automodule:: nnaps.mesa.common_envelope
   :members: apply_ce
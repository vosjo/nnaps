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

If you want to check a model against one of those stability criteria you can use the
:func:`~nnaps.mesa.common_envelope.is_stable` function. For example the code below will ready a compressed mesa model
and check if it is stable using the criterion that a model is unstable if J_div_Jdot_div_P < 10:

.. code-block:: python

    from nnaps.mesa import fileio, common_envelope

    history, _ = fileio.read_compressed_track(<path_to_model.h5>)
    stable = common_envelope.is_stable(history, criterion='J_div_Jdot_div_P', value=10)

    if stable:
        print('The model is stable')
    else:
        print('The model is unstable')


.. _common_envelope:

Common Envelope
---------------

The following formalisms are defined in NNaPS:

========================================= ===================================================== ================================================== =====================
 Name                                      Function                                              Parameters                                         Profile Integration
========================================= ===================================================== ================================================== =====================
:ref:`Iben & Tutukov 1984<iben_tutukov>`  :func:`~nnaps.mesa.common_envelope.iben_tutukov1984`   :math:`\alpha_{\rm CE}`                            No
:ref:`Webbink 1984<webbink>`              :func:`~nnaps.mesa.common_envelope.webbink1984`        :math:`\alpha_{\rm CE}`, :math:`\lambda{\rm CE}`   No
:ref:`Demarco et al 2011<demarko>`        :func:`~nnaps.mesa.common_envelope.demarco2011`        :math:`\alpha_{\rm CE}`, :math:`\lambda_{\rm CE}`  No
:ref:`Dewi & Tauris 2000<dewi_tauris>`    :func:`~nnaps.mesa.common_envelope.dewi_tauris2000`    :math:`\alpha_{\rm CE}`, :math:`\alpha_{\rm TH}`   Yes
========================================= ===================================================== ================================================== =====================

- iben_tutukov1984: `Iben & Tutukov 1984, ApJ, 284, 719 <https://ui.adsabs.harvard.edu/abs/1984ApJ...284..719I/abstract>`_
- webbink1984: `Webbink 1984, ApJ, 277, 355 <https://ui.adsabs.harvard.edu/abs/1984ApJ...277..355W/abstract>`_
- dewi_tauris2000: `Dewi and Tauris 2000, A&A, 360, 1043 <https://ui.adsabs.harvard.edu/abs/2000A%26A...360.1043D/abstract>`_
- demarco2011: `De Marco et al. 2011, MNRAS, 411, 2277 <https://ui.adsabs.harvard.edu/abs/2011MNRAS.411.2277D/abstract>`_

An up to date list of all recognized CE formalisms including possible experimental formalisms can also be obtained with:

.. code-block:: python

    from nnaps.mesa.common_envelope import CE_FORMALISMS
    print(CE_FORMALISMS)


Details of the different formalisms, which parameters are required from MESA for them to work, and which parameters can
be varied are given below.

.. _iben_tutukov:

Iben & Tutukov 1984
^^^^^^^^^^^^^^^^^^^

The CE ejection formalism of `Iben & Tutukov 1984, ApJ, 284, 719 <https://ui.adsabs.harvard.edu/abs/1984ApJ...284..719I/abstract>`_,
one of the earlier CE formalisms using only the alpha parameter. The separation after the CE phase is calculated as
follows:

.. math::

    a_{\rm final} = \alpha \frac{M_{\rm core} \cdot M_2}{M_1^2} a_{\rm init}

Where :math:`M_1` and :math:`M_2` are the donor and companion mass just before the CE starts, :math:`M_{\rm core}` is
the core mass of the donor that remains after the CE is ejected. This is assumed the same as the core mass before the
CE starts. :math:`a_{\rm init}` is the separation just before the CE, :math:`a_{\rm final}` is the final separation and
:math:`\alpha` is the alpha parameter of the formalism.

Required history parameters:
    - star_1_mass
    - star_2_mass
    - he_core_mass
    - binary_separation

.. _webbink:

Webbink 1984
^^^^^^^^^^^^

The CE formalism from `Webbink 1984, ApJ, 277, 355 <https://ui.adsabs.harvard.edu/abs/1984ApJ...277..355W/abstract>`_,
using both the alpha and lambda parameter. The separation after the CE phase is calculated as follows:

.. math::

    a_{\rm final} = \frac{a_{\rm init}\ \alpha\ \lambda\ RL_1\ M_{\rm core}\ M_2} {2\ a_{\rm init}\ M_1\ M_{\rm env} + \alpha\ \lambda\ RL_1\ M_1\ M_2}

Where :math:`M_1` and :math:`M_2` are the donor and companion mass just before the CE starts, :math:`M_{\rm core}` and
:math:`M_{\rm env}` are the core mass and envelope mass of the donor just before the CE starts. :math:`RL_1` is the
Roche lobe radius of the donor at the start of CE. :math:`a_{\rm init}` is the separation at the start of CE,
:math:`a_{\rm final}` is the final separation and :math:`\alpha` and :math:`\gamma` are the alpha and gamma parameters
of this formalism.

Parameters:
    - :math:`\alpha`: the efficiency parameter for the CE formalism
    - :math:`\lambda`: the mass distribution factor of the primary envelope: lambda * Rl = the effective mass-weighted
      mean radius of the envelope at the start of CE.

Required history parameters:
    - star_1_mass
    - star_2_mass
    - he_core_mass
    - binary_separation
    - rl_1

.. _demarko:

Demarco et al 2011
^^^^^^^^^^^^^^^^^^

The CE formalism from `De Marco et al. 2011, MNRAS, 411, 2277 <https://ui.adsabs.harvard.edu/abs/2011MNRAS.411.2277D/abstract>`_,
using an alpha and lambda parameter. This formalism is a slight variation on the formalisms of :ref:`Webbink 1984<webbink>`.
The separation after the CE phase is calculated as follows:

.. math::

    a_{\rm final} = \frac{a_{\rm init}\ \alpha\ \lambda\ RL_1\ M_{\rm core}\ M_2} {M_{\rm env}\ (M_{\rm env} / 2.0 + M_{\rm core})\ a_{\rm init} + \alpha\ \lambda\ RL_1\ M_1\ M_2}

Where :math:`M_1` and :math:`M_2` are the donor and companion mass just before the CE starts, :math:`M_{\rm core}` and
:math:`M_{\rm env}` are the core mass and envelope mass of the donor just before the CE starts. :math:`RL_1` is the
Roche lobe radius of the donor at the start of CE. :math:`a_{\rm init}` is the separation at the start of CE,
:math:`a_{\rm final}` is the final separation and :math:`\alpha` and :math:`\gamma` are the alpha and gamma parameters
of this formalism.

Parameters:
    - :math:`\alpha`: the efficiency parameter for the CE formalism
    - :math:`\lambda`: the mass distribution factor of the primary envelope: lambda * Rl = the effective mass-weighted
      mean radius of the envelope at the start of CE.

Required history parameters:
    - star_1_mass
    - star_2_mass
    - he_core_mass
    - binary_separation
    - rl_1

.. _dewi_tauris:

Dewi & Tauris 2000
^^^^^^^^^^^^^^^^^^

This CE formalism is presented in `Dewi and Tauris 2000, A&A, 360, 1043 <https://ui.adsabs.harvard.edu/abs/2000A%26A...360.1043D/abstract>`_,
based on the idea of obtaining the binding energy by integrating the stellar profile from
`Han et al 1995, MNRAS, 272, 800 <https://ui.adsabs.harvard.edu/abs/1995MNRAS.272..800H/abstract>`_

The CE formalism will integrate over the profile provided. For each cell in the profile it will remove the mass in that
cell and then calculate the change in separation due to the removal of that mass. The integration will continue until
the donor star will stop overfilling it's Roche lobe or until both stars merge. The change in separation in each cell
step is calculated as follows:

.. math::

    da = dm \cdot (\frac{G M_1}{R_c} - \alpha_{\rm TH} U + \frac{\alpha_{\rm CE} G M_2}{2 a}) \frac{2 a^2} {\alpha_{\rm CE} G M_1 M_2}

Parameters:
    - :math:`\alpha_{\rm CE}`: the efficiency of ce
    - :math:`\alpha_{\rm TH}`: the efficiency of binding energy

Required history parameters:
    - star_2_mass
    - binary_separation

Required profile parameters:
    - mass
    - logR
    - logP
    - logRho


If you want to apply the CE formalism yourself on a mesa model you can use the :func:`~nnaps.mesa.common_envelope.apply_ce` function:

.. _hydrogen_handling:

Handling hydrogen atoms
=======================

Servalcat provides options for handling hydrogen atoms during refinement. Because hydrogen atoms are often difficult to observe at typical resolutions, their treatment involves geometric restraints and controlling whether they respond to the experimental data.

.. note::
   By default, hydrogen atoms are used to maintain proper geometry and contribute to structure factor calculations, but their positions are only driven by geometric restraints. To allow their positions to be actively shifted by the experimental data, you must explicitly enable them.

Control keywords
----------------

The following options are available for ``refine_xtal_norefmac`` and ``refine_spa_norefmac``.

.. option:: --hydrogen <all|yes|no>

   Specifies how hydrogen atoms are treated or generated before refinement. This option inherits the behaviour of the ``make hydr`` keyword in Refmac with slight differences.

   * **all** (default): Hydrogen atoms are (re)generated for the model. However, "rotatable" hydrogen atoms (where multiple conformations are geometrically possible, such as hydroxyl hydrogens in Ser/Thr/Tyr) are *not* generated. This prevents incorrect initial placement from introducing artificial steric clashes.
     
     .. note::
        Histidine is a special case under ``all``. While ``HD1`` and ``HE2`` are present in the standard definition, they are **not** generated automatically by ``all`` due to tautomeric ambiguity.
   
   * **yes**: Hydrogen atoms are used exactly as they are present in the input model. No new hydrogens are generated, and existing ones are not removed.
   * **no**: All hydrogen atoms are stripped from the model before refinement.

.. option:: --refine_h

   By default, hydrogen atoms contribute to geometric restraints and the calculation of structure factors, but they are excluded from the gradient calculation against the experimental data (i.e., their positions are not shifted by the data).
   
   Including the ``--refine_h`` flag includes hydrogen atoms in the gradient calculation of the experimental data term, allowing their positions to be refined against the observations.

     .. note::
        Regardless of the settings, **torsion angle restraints** for hydrogen atoms are always active. These restraints are defined in the standard monomer library or user-supplied dictionary files.

.. option:: --hout

   Controls the output of hydrogen atoms. By default, hydrogen atoms are not written to the coordinate file. Specify ``--hout`` to include hydrogen atoms in the output PDB/mmCIF files.

Generating all hydrogen atoms explicitly
-----------------------------------------

To generate all hydrogen atoms including those with multiple potential ideal positions prior to refinement, use the following utility command:

.. code-block:: bash

   servalcat util h_add input.pdb

This command places all hydrogen atoms at one of their ideal positions. It does *not* account for potential steric clashes with neighbouring atoms. Users must inspect the resulting positions and adjust them manually if necessary before proceeding to refinement. For the subsequent refinement, use ``--hydrogen yes --hout``.

Experimental: density-guided hydrogen placement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An experimental option is available to optimise the placement of rotatable hydrogen atoms using experimental data:

.. code-block:: bash

    servalcat util h_add input.pdb --map diffmap_normalized_fofc.mrc --pos nucl

By providing a hydrogen-omit normalised :math:`F_{\rm o} - F_{\rm c}` map via the ``--map`` option, this utility will attempt to place rotatable hydrogen atoms based on the density while simultaneously avoiding steric clashes. The candidates for placement are chosen from the ideal positions defined by the monomer library. Use ``--pos nucl`` (placing hydrogen at nucleus positions) when neutron or cryo-EM data is used.

.. warning::
    This feature is experimental. Users **must** carefully inspect the resulting model. Pay close attention to the end of the log file, where the utility lists any high-density hydrogen candidates that were rejected due to clashes with neighbouring atoms.
    
Tautomers and protonation states
--------------------------------

Servalcat follows the monomer library definitions for tautomers and protonation states. These definitions are generally intended to approximate standard protonation states under typical physiological conditions (e.g., carboxyl groups are deprotonated, and amino groups are protonated).

To use alternative tautomers or protonation states, use the **modification** mechanism. The following modifications are defined within the monomer library:

* ``ASPprot``: Protonated aspartate
* ``GLUprot``: Protonated glutamate
* ``HISprot1``: Histidine tautomer with HE2 (HD1 absent)
* ``HISprot2``: Histidine tautomer with HD1 (HE2 absent)

To apply these modifications to specific residues, use ``MODRES`` records (for PDB files) or the ``_pdbx_struct_mod_residue`` category (for mmCIF files).

We use the unused column (positions 73-80) in `MODRES <https://www.wwpdb.org/documentation/file-format-content/format33/sect3.html#MODRES>`_:

.. code-block:: none

   MODRES      GLU A    7                                                  GLUprot

and we use our own addition ``ccp4_mod_id`` in `_pdbx_struct_mod_residue <https://mmcif.pdb.org/dictionaries/mmcif_ma.dic/Categories/pdbx_struct_mod_residue.html>`_:

.. code-block:: none

   loop_
   _pdbx_struct_mod_residue.id
   _pdbx_struct_mod_residue.auth_asym_id
   _pdbx_struct_mod_residue.auth_seq_id
   _pdbx_struct_mod_residue.PDB_ins_code
   _pdbx_struct_mod_residue.auth_comp_id
   _pdbx_struct_mod_residue.label_comp_id
   _pdbx_struct_mod_residue.parent_comp_id
   _pdbx_struct_mod_residue.details
   _pdbx_struct_mod_residue.ccp4_mod_id
   1 A 7 ? GLU GLU ? ? 'GLUprot'
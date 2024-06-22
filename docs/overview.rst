Overview
========

Servalcat offers various functionalities for refinement and map calculations in crystallography and single particle analysis (SPA).

* SPA
   * refinement (refine_spa_norefmac)
   * sharpened and weighted map calculation (Fo and Fo-Fc)
   * map trimming tool
* crystallography
   * amplitude or intensity based refinement (refine_xtal_norefmac)
   * map calculation from ML parameter estimation (sigmaa)
* others/general
   * REFMAC5 wrapper ("refmacat")
   * geometry optimisation 

The basic usage is:

.. code-block:: console

    $ servalcat <command> <args>

The most common commands are listed below. To see all arguments for a specific command, run:

.. code-block:: console

    $ servalcat <command> -h


Command examples for cryo-EM SPA
--------------------------------

refinement
~~~~~~~~~~
Servalcat performs reciprocal space refinement for single particle analysis. The weighted and sharpened Fo-Fc map is calculated after the refinement. For details please see the reference.
Make a new directory and run:

.. code-block:: console

    $ servalcat refine_spa_norefmac \
     --model input.pdb --resolution 2.5 \
     --halfmaps ../half_map_1.mrc ../half_map_2.mrc \
     --ncycle 10 [--pg C2] \
     [--mask_for_fofc mask.mrc] [-o output_prefix]

Provide unsharpened and unweighted half maps (e.g., from RELION's Refine3D) after ``--halfmaps``.

If map has been symmetrised with a point group, use ``--pg`` to specify the point group symbol along with the asymmetric unit model.
The centre of the box is assumed to be the origin of symmetry. The axis convention follows `RELION <https://relion.readthedocs.io/en/latest/Reference/Conventions.html#symmetry>`_.

Other useful options:
   * ``--ligand lig.cif`` : specify restraint dictionary (.cif) file(s)
   * ``--mask_for_fofc mask.mrc`` : specify mask file for Fo-Fc map calculation
   * ``--jellybody`` : turn on jelly body refinement
   * ``--weight value`` : specify the weight. By default Servalcat determines it from resolution and mask/box ratio
   * ``--keyword_file file`` : specify any refmac keyword file(s) (e.g. prosmart restraint file) Note that not all refmac keywords are supported
   * ``--pixel_size value`` : override pixel size of map

Output files:
   * ``prefix.pdb``: refined model
   * ``prefix_expanded.pdb``: symmetry-expanded version
   * ``prefix_diffmap.mtz``: can be auto-opened with coot. sharpened and weighted Fo map and Fo-Fc map
   * ``prefix_diffmap_normalized_fofc.mrc``: Fo-Fc map normalised within a mask. Look at raw values

Fo-Fc map calculation
~~~~~~~~~~~~~~~~~~~~~
It is crucial to refine individual atomic B values with electron scattering factors for a meaningful Fo-Fc map.
While the ``refine_spa_norefmac`` command calculates the Fo-Fc map (explained above), you can use the fofc command to calculate specific maps, like omit maps.

.. code-block:: console

    $ servalcat fofc \
      --model input.pdb --resolution 2.5 \
      --halfmaps ../half_map_1.mrc ../half_map_2.mrc \
      [--mask mask.mrc] [-o output_prefix] [-B B value]


``-B`` is to calculate weighted maps based on local B estimate. It may be useful for model building in noisy region.

Map trimming
~~~~~~~~~~~~
Maps from single particle analysis are often large due to unnecessary regions outside the molecule. Use trim to save disk space by removing these regions.

.. code-block:: console

    $ servalcat trim \
      --maps postprocess.mrc halfmap1.mrc halfmap2.mrc \
      [--mask mask.mrc] [--model model.pdb] [--padding 10]

Maps specified with ``--maps`` are trimmed. The boundary is decided by ``--mask`` or ``--model`` if mask is not available.
Model(s) are shifted into a new box.
By default new boundary is centred on the original map and cubic, but they can be turned off with ``--noncentered`` and ``--noncubic``.
If you do not want to shift maps and models, specify ``--no_shift`` to keep origin.


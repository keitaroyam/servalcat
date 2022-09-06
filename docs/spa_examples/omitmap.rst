Calculating Fo-Fc omit map
===============================

Here we demonstrate calculation of :math:`F_{\rm o}-F_{\rm c}` omit map, which is the :math:`F_{\rm o}-F_{\rm c}` difference map calculated after removing (omitting) certain atoms. It is useful to show existence of ligand, for example.

You need to prepare 1) unweighted and unsharpened half maps, 2) mask, 3) model files.

Refine the model first!
-----------------------
It is important to refine the model before calculating :math:`F_{\rm o}-F_{\rm c}` map. At least ADP (B-values) should be refined to obtain a meaningful result.
If you want to calculate :math:`F_{\rm o}-F_{\rm c}` map from PDB deposited models, they may have been refined in inappropriate way such as grouped ADP.
Please see :doc:`ChRmine example <chrmine>` for the refinement tutorial.

.. _normalisation-within-mask:

Normalisation within mask
-------------------------
In crystallography, it is common to use "sigma-scaled" map. We often see something like ":math:`mF_{\rm o}-DF_{\rm c}` omit map contoured at 3σ" in the literatures. This σ is standard deviation of map values in the unit cell.

In SPA, we cannot use sigma values calculated in the same way, because box size is arbitrary and everything outside a mask is zero.
Larger box size (more zero-valued pixels) gives smaller sigma value, so it inflates sigma-scaled peak heights.
For details please see section 3.3 of `Yamashita et al. (2021) <https://doi.org/10.1107/S2059798321009475>`_.

In Servalcat, if you give mask file to the program, it calculates maps normalised using sigma calculated within the mask. Please be careful not to see sigma-scaled values shown by graphics programs. For example, you should not look at map values with rmsd unit in Coot.
The raw map values are what you should look at.
In the figure caption I would recommend to write something like ":math:`F_{\rm o}-F_{\rm c}` map is contoured at XXσ (where σ is the standard deviation within the mask)".

Tutorial
---------

Here I use AlF\ :sub:`4`\ \ :sup:`-`\ -ADP–bound state of P4-ATPase flippase (`PDB 6k7k <https://www.rcsb.org/structure/6k7k>`_, `EMD-9937 <https://www.emdataresource.org/EMD-9937>`_) from `Hiraizumi et al. 2019 <https://doi.org/10.1126/science.aay3353>`_ as example.
We need the pdb (or mmcif) file, half maps and mask:
::

    wget https://files.rcsb.org/download/6k7k.pdb
    wget https://ftp.wwpdb.org/pub/emdb/structures/EMD-9937/other/emd_9937_half_map_1.map.gz
    wget https://ftp.wwpdb.org/pub/emdb/structures/EMD-9937/other/emd_9937_half_map_2.map.gz
    wget https://ftp.wwpdb.org/pub/emdb/structures/EMD-9937/masks/emd_9937_msk_1.map

First, refine the model.

.. code-block:: console

    $ servalcat refine_spa \
      --model ../6k7k.pdb \
      --halfmaps ../emd_9937_half_map_1.map.gz ../emd_9937_half_map_2.map.gz \
      --mask_for_fofc  ../emd_9937_msk_1.map \
      --resolution 2.9

Check the refinement result. To calculate omit map properly, overall model quality should be high enough.

Then remove atoms that you want to see in omit map. Here I removed ADP, ALF, and Mg ions (A/1201-1204) using Coot and saved the model as ``refined_omit.pdb``.

Here is the command to calculate :math:`F_{\rm o}-F_{\rm c}` map:

.. code-block:: console

    $ servalcat fofc \
      --model refined_omit.pdb \
      --halfmaps ../emd_9937_half_map_1.map.gz ../emd_9937_half_map_2.map.gz \
      --mask  ../emd_9937_msk_1.map \
      --resolution 2.9 --normalized_map \
      -o diffmap_omit

* ``--halfmaps`` should be unsharpened and unweighted half maps (you used the same half maps for refinement).
* ``--mask`` (not ``--mask_for_fofc``) is used for FSC weighting and normalisation of map values.
* ``--normalized_map`` is to normalise map values within mask.
* ``--resolution`` likewise, it is always good to specify a bit higher resolution than the global one (as determined by the FSC=0.143 criterion).
* ``-o`` is prefix of output file name. Default is ``diffmap``, which would overwrite the file written by ``refine_spa`` job if you run in the same directory.


Check omit map with Coot
~~~~~~~~~~~~~~~~~~~~~~~~
Open ``refined.pdb`` and Auto-open ``diffmap_omit.mtz`` in Coot.
In Display Manager, turn off ``FWT PHWT`` map and adjust contour level of ``DELFWT PHDELWT`` map (this is :math:`F_{\rm o}-F_{\rm c}` map).
Then you see:

.. image:: p4_figs/coot_fofc_omit_4sigma.png
    :align: center
    :scale: 30%

Here at the top, "4.000 e/A^3 (11.87rmsd)" is shown. You may think this is contoured at 11.87σ, but no, this is actually at 4σ. In Coot, a value outside the brackets is a raw map value (ignore e/A^3 or V unit!). See `above <#normalisation-within-mask>`_ also.
Note that this normalisation within mask only happens when ``--mask`` and ``--normalized_map`` are given.

Check omit map with PyMOL
~~~~~~~~~~~~~~~~~~~~~~~~~
PyMOL by default scales maps with sigma (calculated using all pixels) upon reading of map files. It should be turned off before reading map files. So first start PyMOL with the model file only,

.. code-block:: console

    $ pymol refined.pdb

and then turn off normalisation in PyMOL:
::

    set normalize_ccp4_maps, off
    load diffmap_omit_normalized_fofc.mrc
    isomesh msh_fofc, diffmap_omit_normalized_fofc, 4

You see:

.. image:: p4_figs/pymol_fofc_omit_4sigma.png
    :align: center
    :scale: 40%

Again, this is :math:`F_{\rm o}-F_{\rm c}` omit map contoured at 4σ (where σ is the standard deviation within the mask).

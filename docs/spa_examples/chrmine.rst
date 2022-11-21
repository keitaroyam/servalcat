Refinement of ChRmine structure
===============================

Here we demonstrate the atomic model refinement with *C*\ 3 symmetry using ChRmine (`Kishi et al. 2020 <http://dx.doi.org/10.1016/j.cell.2022.01.007>`_, `PDB 7w9w <https://www.rcsb.org/structure/7w9w>`_, `EMD-32377 <https://www.emdataresource.org/EMD-32377>`_).
We need the pdb (or mmcif) file, half maps and mask:
::

    wget https://files.rcsb.org/download/7w9w.pdb
    wget https://ftp.wwpdb.org/pub/emdb/structures/EMD-32377/other/emd_32377_half_map_1.map.gz
    wget https://ftp.wwpdb.org/pub/emdb/structures/EMD-32377/other/emd_32377_half_map_2.map.gz
    wget https://ftp.wwpdb.org/pub/emdb/structures/EMD-32377/masks/emd_32377_msk_1.map

.. note::
    Half maps must be unsharpened and unweighted. In this case ones from Refine3D job of RELION are used. Mask file is only used for map calculation after the refinement (thus it does not affect the refinement).


**In this example please use Refmac5 from CCP4 8.0.007 (Refmac5.8.0403 or newer) and Servalcat from CCP-EM 1.7 (Servalcat 0.3.0 or newer).**

Run refinement from command-line
--------------------------------
Servalcat Refmac pipeline is available via refine_spa subcommand. Run following command in *a new directory* as this command creates files with fixed names.

.. code-block:: console

    $ servalcat refine_spa \
      --model ../7w9w.pdb \
      --halfmaps ../emd_32377_half_map_1.map.gz ../emd_32377_half_map_2.map.gz \
      --mask_for_fofc  ../emd_32377_msk_1.map \
      --pg C3 --resolution 1.95 --cross_validation 

* ``--halfmaps`` should be unsharpened and unweighted half maps.
* ``--mask_for_fofc`` is only used in map calculation after the refinement. It does not affect the refinement.
* ``--cross_validation`` is to run cross-validation using half maps as described in `Brown et al. 2015 <https://doi.org/10.1107/S1399004714021683>`_.
* ``--pg C3`` is specified because the map is symmetrised with *C*\ 3 symmetry. In this case ``--model`` must be an asymmetric unit. The orientation of group and origin follow `RELION's convention <https://relion.readthedocs.io/en/latest/Reference/Conventions.html#symmetry>`_: 3-fold is along z-axis through the centre of the box.
* ``--resolution`` is the resolution used in refinement and map calculation. It is always good to specify a bit higher value than the global one (as determined by the FSC=0.143 criterion), because local resolution can be higher. Here the global resolution was 2.02 Å so I put 1.95 Å.

.. note::
    If the pixel size in map file header is wrong, you can specify the correct pixel size using ``--pixel_size`` option. Note that this affects all input map and mask files, but not for input model. Model should overlap with map with correct the pixel size, and needs to be fixed before refinement if the model is fitted to a map with wrong pixel size.

In case you want to know what Servalcat does in the pipeline:

#. Expand model with *C*\ 3 symmetry (written as input_model_expanded.pdb).
#. Create a mask (mask_from_model.ccp4) around the model with 3 Å radius. A radius can be changed using ``--mask_radius``.
#. Trim half maps using the mask and shift the model to the new origin (shifted.pdb).
#. FFT maps after sharpen-mask-unsharpen procedure and writes Fourier coefficients to mtz file (masked_fs_obs.mtz).
#. Let Refmac5 refine shifted.pdb against masked_fs_obs.mtz, and write shifted_refined.pdb. Refmac instructions are:

    * ``make hydr all`` to generate hydrogen atoms; can be changed using ``--hydrogen``
    * ``make hout no`` not to write hydrogen atoms in the output; specify ``--hout`` if you want to have hydrogen in the output model
    * ``solvent no`` not to consider bulk solvent effect
    * ``scale lssc isot`` to only do isotropic overall scaling (no anisotropic scaling)
    * ``source em mb`` to calculate atomic scattering factor using Mott-Bethe formula
    * ``ncycle 10`` to run 10 conjugate gradient cycles; can be changed using ``--ncycle``
    * ``weight auto 8.31e-01`` the weight is automatically determined by Servalcat using mask volume ratio and effective resolution
    * ``ncsr local`` to use local NCS restraints if multiple chains with similar sequences exist
    * ``@ncsc_shifted.txt`` to tell symmetry operators of point group symmetry specified by ``--pg``.

#. Run cross validation

    #. Shake the model with rms of 0.3 Å; can be changed using ``--shake_radius``
    #. Refine the model (shifted_refined.pdb) against half map 1, and write shifted_refined_shaken_refined.pdb.

#. Shift back the models to the original position
#. Expand the final model with symmetry (refined_expanded.pdb)
#. Calculate map-model FSC
#. Calculate sharpened and weighted Fo and Fo-Fc maps (diffmap.mtz, diffmap_normalized_fo.mrc, diffmap_normalized_fofc.mrc)
#. Show final summary

Final summary is like this:

.. code-block:: none

    =============================================================================
    * Final Summary *

    Rmsd from ideal
      bond lengths: 0.0062 A
      bond  angles: 1.315 deg

    Map-model FSCaverages (at 1.95 A):
     FSCaverage(full) =  0.8399
    Cross-validated map-model FSCaverages:
     FSCaverage(half1)=  0.7856
     FSCaverage(half2)=  0.7614
     Run loggraph refined_fsc.log to see plots

    ADP statistics
     Chain A (2400 atoms) min= 20.9 median= 48.1 max=189.7 A^2

    Weight used: 0.830999970
                 If you want to change the weight, give larger (looser restraints)
                 or smaller (tighter) value to --weight_auto_scale=.

    Open refined model and diffmap.mtz with COOT:
    coot --script refined_coot.py

    List Fo-Fc map peaks in the ASU:
    servalcat util map_peaks --map diffmap_normalized_fofc.mrc --model refined.pdb --abs_level 4.0
    =============================================================================

.. _chrmine-check-fsc:

Check FSC
~~~~~~~~~
You can use loggraph command from CCP4 to see map-model FSC vs resolution.

.. code-block:: console

    $ loggraph refined_fsc.log

.. image:: chrmine_figs/refined_fsc_1.png
    :align: center
    :scale: 40%

Note

* In loggraph, x-axis scale is 1/d^2, while in SPA usually 1/d scale is used.
* Sharpened-masked-unsharpened half maps are used for half map FSC (FSC_half) with the mask used in the refinement. Currently phase randomisation is not performed.
* FSC_full_sqrt is the estimated correlation between full map and true map: :math:`\sqrt{2{\rm FSC_{half}}/(1+{\rm FSC_{half}})}`. If FSC(full,model) is higher than this, it may indicate overfitting (see `Nicholls et al. 2018 <https://doi.org/10.1107/S2059798318007313>`_).
* FSC curves are calculated up to Nyquist resolution

refined_fsc.json contains the same data. If you want to use external programs to plot FSC (such as R or MS Excel), you can convert it to csv file:

.. code-block:: console

    $ servalcat util json2csv refined_fsc.json

Check maps and model
~~~~~~~~~~~~~~~~~~~~
Let us open the refined model and maps with COOT:

.. code-block:: console

    $ coot --script refined_coot.py

Do not look at contour levels in "rmsd" (so-called sigma). In SPA, the sigma-level is useless, because box size is arbitrary and volumes outside the mask are all zero that leads to underestimate of sigma value.
In this example we gave a mask file (with ``--mask_for_fofc``) so these maps are normalised within the mask. So raw map values can be considered "sigma level" in usual (crystallographic) sense. In COOT raw map values are shown with e/A^3 or V unit (these units are not right). Again, do not see values with rmsd unit in case of SPA!

You may find something interesting from the Fo-Fc map. Below is putative hydrogen densities (shown at 3 sigma level). Note that the map is calculated without hydrogen contribution (thus hydrogen omit Fo-Fc map) unless ``--hout`` is specified.

.. image:: chrmine_figs/coot_113-fs8.png
    :align: center
    :scale: 40%

In other graphics programs such as Chimera or PyMOL, open diffmap_normalized_fo.mrc and diffmap_normalized_fofc.mrc for Fo and Fo-Fc maps, respectively. PyMOL by default scales maps by their "sigma", so you should run ``set normalize_ccp4_maps, off`` before opening mrc files.

Run Molprobity
~~~~~~~~~~~~~~
If you want to check Ramachandran plots, rotamer outliers, clash scores etc for the table of paper, you can run

.. code-block:: console

    $ molprobity.molprobity refined_expanded.pdb nqh=false

It writes molprobity_coot.py which can be opened with COOT (from Calculate - Run Script...) to see "ToDo list". Note that the outliers are not always wrong - you should check them with density.

.. code-block:: console

    $ coot --script refined_coot.py --script molprobity_coot.py

Run refinement from GUI
-----------------------
#. Start ``ccpem`` and push "Refmac Servalcat" button or run ``ccpem-refmac`` command.
#. Fill Input model, Resolution, Half map 1 & 2, and Mask for Fo-Fc map. For others see:

    .. image:: chrmine_figs/ccpem_input-fs8.png
        :align: center
        :scale: 40%

#. Push Run button
#. Full Refmac5 log is shown
#. You can see plots in Results panel and open files with external programs in Launcher panel.

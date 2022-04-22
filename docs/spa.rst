Single particle analysis (SPA)
==============================

Half maps
---------
TBD

Point group symmetry
--------------------
In SPA maps may have point group symmetry: :math:`C_n, D_n, T, O,` or :math:`I`. In such case atomic model must follow the symmetry. Our recommendation is to refine a model in the asymmetric unit (ASU) with symmetry constraints. In crystallography we always refine ASU model and refinement programs internally take symmetry copies into account. We want to do the same thing in SPA.

The ASU model can be deposited to PDB together with symmetry annotations (e.g. `7a4m <https://www.rcsb.org/structure/7a4m>`_, `7a5v <https://www.rcsb.org/structure/7a5v>`_, `7w9w <https://www.rcsb.org/structure/7w9w>`_). Both ASU model and biological assembly are available at the PDB website. Strictly speaking, biological assembly is not always the same as symmetry in reconstruction. I will explain this later.

Symmetry information (list of rotation matrix and translation vector) is described in `MTRIX <https://www.wwpdb.org/documentation/file-format-content/format33/sect8.html#MTRIXn>`_ header of PDB file or in `_struct_ncs_oper <https://mmcif.wwpdb.org/dictionaries/mmcif_std.dic/Categories/struct_ncs_oper.html>`_ of mmCIF file.

For example, in 7w9w.pdb:
::

    MTRIX1   1  1.000000  0.000000  0.000000        0.00000    1
    MTRIX2   1  0.000000  1.000000  0.000000        0.00000    1
    MTRIX3   1  0.000000  0.000000  1.000000        0.00000    1
    MTRIX1   2 -0.500000 -0.866025  0.000000      180.75898
    MTRIX2   2  0.866025 -0.500000  0.000000       48.43422
    MTRIX3   2  0.000000  0.000000  1.000000        0.00000
    MTRIX1   3 -0.500000  0.866025  0.000000       48.43422
    MTRIX2   3 -0.866025 -0.500000  0.000000      180.75898
    MTRIX3   3  0.000000  0.000000  1.000000        0.00000

in 7w9w.cif:
::

    loop_
    _struct_ncs_oper.id
    _struct_ncs_oper.code
    _struct_ncs_oper.matrix[1][1]
    _struct_ncs_oper.matrix[1][2]
    _struct_ncs_oper.matrix[1][3]
    _struct_ncs_oper.vector[1]
    _struct_ncs_oper.matrix[2][1]
    _struct_ncs_oper.matrix[2][2]
    _struct_ncs_oper.matrix[2][3]
    _struct_ncs_oper.vector[2]
    _struct_ncs_oper.matrix[3][1]
    _struct_ncs_oper.matrix[3][2]
    _struct_ncs_oper.matrix[3][3]
    _struct_ncs_oper.vector[3]
    _struct_ncs_oper.details
    1 given    1    0            0 0          0            1    0 0          0 0 1 0 ?
    2 generate -0.5 -0.866025404 0 180.758982 0.866025404  -0.5 0 48.4342232 0 0 1 0 ?
    3 generate -0.5 0.866025404  0 48.4342232 -0.866025404 -0.5 0 180.758982 0 0 1 0 ?

Using this information, symmetry-expanded model can be generated. Various programs can do this:

.. code-block:: console

    $ gemmi convert --expand-ncs=new 7w9w.pdb 7w9w_expanded.pdb
    $ servalcat util expand --model 7w9w.pdb
    $ phenix.pdb.mtrix_reconstruction 7w9w.pdb

For the refinement with point group symmetry, please see :doc:`ChRmine example <spa_examples/chrmine>`.

Helical symmetry
----------------
In helical reconstruction, axial symmetry (:math:`C_n` or :math:`D_n`), twist (in degree), and rise (in Å) parameters define the symmetry. 

For the refinement with helical symmetry, please see :doc:`amyloid-β 42 example <spa_examples/ab42>`.

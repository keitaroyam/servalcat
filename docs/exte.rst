External restraints
===================

.. warning::
   **Experimental Feature:** Defining external restraints directly via JSON
   configuration files is currently under active development. The JSON schema,
   supported keys, and default behaviours are subject to change in future
   releases.

This guide explains how to define and apply external geometry restraints using
JSON files in Servalcat's refinement programs (``refine_xtal_norefmac``,
``refine_spa_norefmac``, and ``refine_geom``).

In addition to :doc:`Refmac-compatible keyword files <refmac_keywords>`
(``--keyword_file`` / ``--keywords``), Servalcat can read external restraints
from JSON files specified under ``refine.exte_files`` in the refinement YAML
configuration file (passed via ``--config``):

.. code-block:: yaml

   refine:
     exte_files:
       - "exte.json"

Some restraint types, such as centroid distance (``cdist``) and centroid angle
(``cangl``) restraints, can only be defined using this JSON format.

File structure
--------------

An external restraint JSON file contains a top-level JSON array (list) of
objects processed sequentially. Each object in the list can contain a
``"defaults"`` dictionary (updating processing options for subsequent entries in
the same file) and/or a restraint definition consisting of ``"rest_type"`` and
``"restr"``:

.. code-block:: json

   [
     {
       "defaults": {
         "ignore_undefined": true,
         "ignore_hydrogens": true,
         "scale_sigma_dist": 1.0
       }
     },
     {
       "rest_type": "dist",
       "restr": {
         "value": 1.54,
         "sigma": 0.02,
         "specs": [
           [{"chain": "A", "resi": 10, "names": ["C1"]}],
           [{"chain": "A", "resi": 10, "names": ["C2"]}]
         ]
       }
     }
   ]

Global defaults configuration
-----------------------------

Processing options can be set at the beginning of a JSON array or updated at
any point within the array via a ``"defaults"`` block:

- Multiple ``"defaults"`` blocks can appear throughout the JSON list.
- Updated settings apply to the current entry and all subsequent restraint
  definitions in the same file until overridden by another ``"defaults"`` block.
- Defaults are reset to their built-in initial values at the start of each
  external restraint file.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``symall_block``
     - ``false``
     - Enable symmetry image search (crystallographic symmetry and strict NCS
       defined in the model header via ``MTRIX`` or ``_struct_ncs_oper``) by
       default for subsequent ``dist``, ``cdist``, ``cangl``, and ``inte``
       restraints.
   * - ``exclude_self_block``
     - ``false``
     - When symmetry search is active, if ``true``, restrict the search to
       symmetry-related copies only (excluding the original / same copy); if
       ``false``, consider both the same and symmetry-related copies and choose
       the nearest image.
   * - ``type_default``
     - ``2``
     - Default restraint ``type`` (``0``, ``1``, or ``2``) for ``dist``
       restraints when ``type`` is not specified in ``restr``.
   * - ``alpha_default``
     - ``1.0``
     - Default ``alpha`` shape parameter for the robust loss function of
       `Barron (2019) <https://arxiv.org/abs/1701.03077>`_ in ``dist``
       restraints (effective when ``type`` is ``2``).
   * - ``scale_sigma_dist``
     - ``1.0``
     - Divisor applied to ``sigma`` values of ``dist`` restraints
       (effective :math:`\sigma = \texttt{sigma} / \texttt{scale\_sigma\_dist}`).
   * - ``scale_sigma_cdist``
     - ``1.0``
     - Divisor applied to ``sigma`` values of ``cdist`` restraints.
   * - ``scale_sigma_angl``
     - ``1.0``
     - Divisor applied to ``sigma`` values of ``angl`` restraints.
   * - ``scale_sigma_cangl``
     - ``1.0``
     - Divisor applied to ``sigma`` values of ``cangl`` restraints.
   * - ``scale_sigma_tors``
     - ``1.0``
     - Divisor applied to ``sigma`` values of ``tors`` restraints.
   * - ``scale_sigma_chir``
     - ``1.0``
     - Divisor applied to ``sigma`` values of ``chir`` restraints.
   * - ``scale_sigma_plan``
     - ``1.0``
     - Divisor applied to ``sigma`` values of ``plan`` restraints.
   * - ``scale_sigma_inte``
     - ``1.0``
     - Divisor applied to explicitly specified ``smin`` and ``smax`` values of
       ``inte`` restraints.
   * - ``sigma_min_loc``
     - ``0.0``
     - Minimum allowed ``sigma`` value (after ``scale_sigma_dist`` scaling) for
       ``dist`` restraints.
   * - ``sigma_max_loc``
     - ``100.0``
     - Maximum allowed ``sigma`` value (after ``scale_sigma_dist`` scaling) for
       ``dist`` restraints.
   * - ``ignore_undefined``
     - ``false``
     - If ``true``, log a warning and skip restraints referencing atoms that do
       not exist in the structure. If ``false``, raise a ``RuntimeError``.
   * - ``ignore_hydrogens``
     - ``true``
     - If ``true``, omit hydrogen atoms when resolving explicit atom
       specifications, and skip ``dist``, ``angl``, ``tors``, and ``inte``
       restraints that reference hydrogen atoms.
   * - ``dist_min_external``
     - :math:`-\infty`
     - Exclusive lower bound on target ``value`` for ``dist`` restraints
       (restraints with ``value <= dist_min_external`` are ignored).
   * - ``dist_max_external``
     - :math:`+\infty`
     - Exclusive upper bound on target ``value`` for ``dist`` restraints
       (restraints with ``value >= dist_max_external`` are ignored).

.. note::
   The keys ``ext_verbose`` (default ``false``), ``use_atoms`` (default
   ``"a"``), and ``prefix_ch`` (default ``" "``) are also accepted in
   ``defaults`` for compatibility with internal Refmac keyword translation, but
   currently have no effect.

Atom specifications (``specs``)
-------------------------------

The ``specs`` field inside ``restr`` is a list of lists of atom-selection
objects (``[[spec, ...], ...]``):

- For grouped restraints (``cdist``, ``cangl``, and ``stac``), each element of
  the outer list ``specs[i]`` defines **one atom group** (a centroid or a
  plane) and may contain one or more atom-selection objects.
- For all other restraint types (``dist``, ``angl``, ``tors``, ``chir``,
  ``plan``, ``inte``, and ``harm``), each element of the outer list ``specs[i]``
  **must** contain exactly one atom-selection object (``len(specs[i]) == 1``).

Two atom-selection object formats are supported: **explicit atom selection**
and **residue-range selection**.

Explicit atom selection
~~~~~~~~~~~~~~~~~~~~~~~

Selects specific atom(s) by chain, residue number, and atom name(s):

.. code-block:: json

   {"chain": "A", "resi": 10, "ins": " ", "names": ["CA"], "alt": "A"}

.. list-table::
   :widths: 15 15 15 55
   :header-rows: 1

   * - Field
     - Type
     - Required
     - Description
   * - ``chain``
     - string
     - Yes
     - Chain identifier (e.g., ``"A"``).
   * - ``resi``
     - integer
     - Yes
     - Residue sequence number.
   * - ``names``
     - list of strings
     - Yes
     - Atom name(s) within the residue (e.g., ``["CA"]`` or
       ``["OD1", "OD2"]``).
   * - ``ins``
     - string
     - No
     - Insertion code (default ``" "`` for none).
   * - ``alt``
     - string
     - No
     - Alternative conformation identifier (e.g., ``"A"``). Omit when the atom
       has no alternative location indicator.
   * - ``symm``
     - boolean
     - No
     - Per-atom symmetry flag (crystal symmetry and strict NCS) used by
       ``dist``, ``angl``, and ``inte`` restraints (default ``false``).

Residue-range selection
~~~~~~~~~~~~~~~~~~~~~~~

When an atom-selection object contains the ``ifirst`` key, it selects atoms
across a range of residue sequence numbers in a chain. This format is intended
for non-grouped restraints such as ``harm``:

.. code-block:: json

   {"chain": "A", "ifirst": 10, "ilast": 25, "atom": "CA"}

.. list-table::
   :widths: 15 20 15 50
   :header-rows: 1

   * - Field
     - Type
     - Required
     - Description
   * - ``chain``
     - string
     - Yes
     - Chain identifier.
   * - ``ifirst``
     - integer or ``null``
     - Yes
     - Starting residue sequence number (inclusive), or ``null`` for no lower
       limit.
   * - ``ilast``
     - integer or ``null``
     - Yes
     - Ending residue sequence number (inclusive), or ``null`` for no upper
       limit.
   * - ``atom``
     - string
     - No
     - Atom name to select in each residue, or ``"*"`` (default) to select all
       atoms in the residue range. Note that ``ignore_hydrogens`` is not applied
       to residue-range selections.

Restraint types
---------------

Each restraint definition object specifies ``rest_type`` and a ``restr``
dictionary whose fields depend on the restraint type:

.. list-table::
   :widths: 15 35 25 25
   :header-rows: 1

   * - ``rest_type``
     - Description
     - Outer ``specs`` length
     - Atoms resolved
   * - ``dist``
     - Bond / interatomic distance
     - 2
     - 1 atom per spec (2 total)
   * - ``cdist``
     - Centroid-to-centroid distance
     - 2
     - 1+ atoms per group (2 centroids)
   * - ``angl``
     - Three-atom bond angle
     - 3
     - 1 atom per spec (3 total)
   * - ``cangl``
     - Three-centroid angle
     - 3
     - 1+ atoms per group (3 centroids)
   * - ``tors``
     - Four-atom torsion (dihedral) angle
     - 4
     - 1 atom per spec (4 total)
   * - ``chir``
     - Chiral volume
     - 4
     - 1 atom per spec (4 total)
   * - ``plan``
     - Planarity
     - 1+
     - 3+ atoms total
   * - ``inte``
     - Flat-bottom distance interval
     - 2
     - 1 atom per spec (2 total)
   * - ``stac``
     - Plane-to-plane stacking
     - 2
     - 3+ atoms per group (2 planes)
   * - ``harm``
     - Positional harmonic restraint
     - 1+
     - 1+ atoms total

.. note::
   ``spec`` (special position restraint) also appears as an internal key in
   ``exte.py``, but is not functional via external restraint JSON files.
   Servalcat automatically detects atoms on special positions (under crystal
   symmetry or strict NCS) and applies position and anisotropic ADP
   constraints.

Distance restraints (``dist``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrains the distance between two atoms.

.. list-table::
   :widths: 15 15 15 55
   :header-rows: 1

   * - Field
     - Type
     - Required
     - Description
   * - ``value``
     - float
     - Yes
     - Target distance in Å. Skipped unless
       ``dist_min_external < value < dist_max_external``.
   * - ``sigma``
     - float
     - Yes
     - Standard deviation in Å (divided by ``scale_sigma_dist`` and clamped to
       ``[sigma_min_loc, sigma_max_loc]``).
   * - ``specs``
     - list
     - Yes
     - Two single-element atom spec lists (each resolving to 1 atom).
   * - ``type``
     - integer
     - No
     - Restraint type (``0``, ``1``, or ``2``; default ``type_default``):

       - ``0``: Override existing bond restraints for the same atom pair and
         register the pair as bonded (excluding non-bonded VDW repulsion). Uses
         harmonic loss (:math:`\alpha = 2`).
       - ``1``: Add to existing bond restraints for the same atom pair (if
         multiple ideal values exist, the closest value to the current model
         distance is chosen in each cycle; overrides ``type = 2``) and register
         as bonded. Uses harmonic loss (:math:`\alpha = 2`).
       - ``2``: External distance restraint using the generalized loss function
         of `Barron (2019) <https://arxiv.org/abs/1701.03077>`_ controlled by
         ``alpha``. Ignored if a ``type = 0`` or ``type = 1`` bond is already
         defined for the same atom pair. Reported under ``External distances``.
   * - ``alpha``
     - float
     - No
     - Shape parameter :math:`\alpha` for Barron's robust loss when
       ``type == 2`` (default ``alpha_default``). Common values include ``2.0``
       (least-squares / harmonic), ``1.0`` (smooth :math:`L_1`-:math:`L_2`),
       ``0.0`` (Cauchy / Lorentzian), ``-2.0`` (Geman-McClure), and values
       ``< -1000`` (Welsch).
   * - ``symm``
     - boolean
     - No
     - Enable symmetry image search (crystal symmetry and strict NCS) for this
       pair (default ``symall_block``). Also enabled if ``"symm": true`` is set
       on either atom specification.

**Example:**

.. code-block:: json

   {
     "rest_type": "dist",
     "restr": {
       "value": 2.8,
       "sigma": 0.1,
       "type": 2,
       "alpha": 1.0,
       "symm": false,
       "specs": [
         [{"chain": "A", "resi": 45, "names": ["O"]}],
         [{"chain": "B", "resi": 61, "names": ["N"]}]
       ]
     }
   }

Centroid distance restraints (``cdist``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrains the distance between the geometric centroids of two atom groups. Within
each group, all selected atoms are weighted equally (:math:`w = 1/N`). This is
useful, for example, when restraining metal coordination to the midpoint of a
carboxylate group.

.. list-table::
   :widths: 15 15 15 55
   :header-rows: 1

   * - Field
     - Type
     - Required
     - Description
   * - ``value``
     - float
     - Yes
     - Target distance between the two centroids in Å.
   * - ``sigma``
     - float
     - Yes
     - Standard deviation in Å (divided by ``scale_sigma_cdist``).
   * - ``specs``
     - list
     - Yes
     - Exactly two atom-group lists, each resolving to one or more atoms.
   * - ``alpha``
     - float
     - No
     - Barron (2019) robust loss parameter :math:`\alpha` (default ``2.0``,
       i.e., harmonic least-squares; note that ``alpha_default`` does not apply
       to ``cdist``).
   * - ``symm``
     - boolean
     - No
     - If ``true`` (default ``symall_block``), atoms within each multi-atom
       centroid group are mapped to their nearest symmetry image relative to the
       first atom of the first group.

**Example:**

.. code-block:: json

   {
     "rest_type": "cdist",
     "restr": {
       "value": 2.15,
       "sigma": 0.08,
       "specs": [
         [{"chain": "A", "resi": 1090, "names": ["CA"]}],
         [{"chain": "A", "resi": 65, "names": ["OD1", "OD2"]}]
       ]
     }
   }

Angle restraints (``angl``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrains the bond angle formed by three atoms, where the **second** atom is the
vertex (central atom).

.. list-table::
   :widths: 15 15 15 55
   :header-rows: 1

   * - Field
     - Type
     - Required
     - Description
   * - ``value``
     - float
     - Yes
     - Target angle in degrees (normalized to :math:`[0^\circ, 180^\circ]`).
   * - ``sigma``
     - float
     - Yes
     - Standard deviation in degrees (divided by ``scale_sigma_angl``).
   * - ``specs``
     - list
     - Yes
     - Three single-element atom spec lists (atom 1 -- atom 2 -- atom 3, with
       atom 2 at the vertex). To involve a symmetry-related outer atom, set
       ``"symm": true`` inside ``specs[0][0]`` and/or ``specs[2][0]``, which
       selects the nearest symmetry-related image (excluding the same copy)
       relative to the vertex atom.

**Example:**

.. code-block:: json

   {
     "rest_type": "angl",
     "restr": {
       "value": 109.5,
       "sigma": 2.0,
       "specs": [
         [{"chain": "A", "resi": 10, "names": ["CA"]}],
         [{"chain": "A", "resi": 10, "names": ["C"]}],
         [{"chain": "A", "resi": 11, "names": ["N"]}]
       ]
     }
   }

Centroid angle restraints (``cangl``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrains the angle formed by three atom-group centroids, where the **second**
centroid is the vertex. Within each group, all selected atoms receive equal
weights (:math:`w = 1/N`).

.. list-table::
   :widths: 15 15 15 55
   :header-rows: 1

   * - Field
     - Type
     - Required
     - Description
   * - ``value``
     - float
     - Yes
     - Target angle between the three centroids in degrees.
   * - ``sigma``
     - float
     - Yes
     - Standard deviation in degrees (divided by ``scale_sigma_cangl``).
   * - ``specs``
     - list
     - Yes
     - Exactly three atom-group lists, each resolving to one or more atoms
       (centroid 2 is the vertex).
   * - ``symm``
     - boolean
     - No
     - If ``true`` (default ``symall_block``), atoms within each multi-atom
       centroid group are mapped to their nearest symmetry image relative to the
       first atom of the first group.

**Example:**

.. code-block:: json

   {
     "rest_type": "cangl",
     "restr": {
       "value": 180.0,
       "sigma": 1.0,
       "specs": [
         [{"chain": "A", "resi": 65, "names": ["OD1", "OD2"]}],
         [{"chain": "A", "resi": 1090, "names": ["CA"]}],
         [{"chain": "A", "resi": 72, "names": ["OE1", "OE2"]}]
       ]
     }
   }

Torsion angle restraints (``tors``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrains the dihedral (torsion) angle defined by four sequential atoms (with
periodicity fixed to ``1``).

.. list-table::
   :widths: 15 15 15 55
   :header-rows: 1

   * - Field
     - Type
     - Required
     - Description
   * - ``value``
     - float
     - Yes
     - Target torsion angle in degrees.
   * - ``sigma``
     - float
     - Yes
     - Standard deviation in degrees (divided by ``scale_sigma_tors``).
   * - ``specs``
     - list
     - Yes
     - Four single-element atom spec lists (each resolving to 1 atom).

**Example:**

.. code-block:: json

   {
     "rest_type": "tors",
     "restr": {
       "value": -60.0,
       "sigma": 20.0,
       "specs": [
         [{"chain": "A", "resi": 10, "names": ["CA"]}],
         [{"chain": "A", "resi": 10, "names": ["C"]}],
         [{"chain": "A", "resi": 11, "names": ["N"]}],
         [{"chain": "A", "resi": 11, "names": ["CA"]}]
       ]
     }
   }

Chirality restraints (``chir``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrains the signed chiral volume around a chiral centre. For centre atom
:math:`\mathbf{x}_c` and neighbour atoms :math:`\mathbf{x}_1, \mathbf{x}_2,
\mathbf{x}_3`, the chiral volume is :math:`((\mathbf{x}_1 - \mathbf{x}_c)
\times (\mathbf{x}_2 - \mathbf{x}_c)) \cdot (\mathbf{x}_3 - \mathbf{x}_c)`.

.. list-table::
   :widths: 15 15 15 55
   :header-rows: 1

   * - Field
     - Type
     - Required
     - Description
   * - ``value``
     - float
     - Yes
     - Signed target chiral volume in :math:`\text{\AA}^3`.
   * - ``sigma``
     - float
     - Yes
     - Standard deviation in :math:`\text{\AA}^3` (divided by
       ``scale_sigma_chir``).
   * - ``specs``
     - list
     - Yes
     - Four single-element atom spec lists (each resolving to 1 atom): the
       chiral centre first, followed by the three bonded neighbour atoms. If any
       of the four atoms is a hydrogen atom, ``ignore_hydrogens`` must be set to
       ``false``.

**Example:**

.. code-block:: json

   {
     "rest_type": "chir",
     "restr": {
       "value": 2.52,
       "sigma": 0.2,
       "specs": [
         [{"chain": "A", "resi": 10, "names": ["CA"]}],
         [{"chain": "A", "resi": 10, "names": ["N"]}],
         [{"chain": "A", "resi": 10, "names": ["C"]}],
         [{"chain": "A", "resi": 10, "names": ["CB"]}]
       ]
     }
   }

Planarity restraints (``plan``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrains a group of three or more atoms to lie on a common least-squares plane.

.. list-table::
   :widths: 15 15 15 55
   :header-rows: 1

   * - Field
     - Type
     - Required
     - Description
   * - ``sigma``
     - float
     - Yes
     - Out-of-plane standard deviation in Å (divided by ``scale_sigma_plan``).
   * - ``specs``
     - list
     - Yes
     - One or more single-element atom spec lists resolving to at least three
       atoms in total (either via multiple atom names in ``names`` within one
       residue or across multiple residue specs).

**Example:**

.. code-block:: json

   {
     "rest_type": "plan",
     "restr": {
       "sigma": 0.02,
       "specs": [
         [{"chain": "A", "resi": 10, "names": ["CA", "C", "O"]}],
         [{"chain": "A", "resi": 11, "names": ["N", "CA"]}]
       ]
     }
   }

Interval restraints (``inte``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Applies a flat-bottom harmonic distance restraint between two atoms: no penalty
is applied when the interatomic distance :math:`d` lies within
:math:`[\texttt{dmin}, \texttt{dmax}]`, while harmonic penalties with standard
deviations ``smin`` and ``smax`` are applied when :math:`d < \texttt{dmin}` or
:math:`d > \texttt{dmax}`, respectively.

.. list-table::
   :widths: 15 20 15 50
   :header-rows: 1

   * - Field
     - Type
     - Required
     - Description
   * - ``dmin``
     - float
     - Conditional
     - Lower distance bound in Å. At least one of ``dmin`` or ``dmax`` must be
       provided; if only one is specified, the other is set equal to it.
   * - ``dmax``
     - float
     - Conditional
     - Upper distance bound in Å.
   * - ``smin``
     - float
     - No
     - Standard deviation in Å when :math:`d < \texttt{dmin}`. If both ``smin``
       and ``smax`` are omitted, both default to ``0.05`` Å (unscaled). If at
       least one of ``smin`` or ``smax`` is provided, any omitted one is set
       equal to the other and both are divided by ``scale_sigma_inte``.
   * - ``smax``
     - float
     - No
     - Standard deviation in Å when :math:`d > \texttt{dmax}`.
   * - ``symm``
     - boolean
     - No
     - Enable symmetry image search (crystal symmetry and strict NCS; default
       ``symall_block``). Also enabled if ``"symm": true`` is set on either
       atom specification.
   * - ``specs``
     - list
     - Yes
     - Two single-element atom spec lists (each resolving to 1 atom).

**Example:**

.. code-block:: json

   {
     "rest_type": "inte",
     "restr": {
       "dmin": 2.6,
       "dmax": 3.2,
       "smin": 0.05,
       "smax": 0.1,
       "specs": [
         [{"chain": "A", "resi": 2, "names": ["O"]}],
         [{"chain": "A", "resi": 4, "names": ["N"]}]
       ]
     }
   }

Stacking restraints (``stac``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Restrains the relative orientation (and optionally plane-to-plane separation) of
two planar atom groups, such as stacked nucleobases or aromatic rings.

.. list-table::
   :widths: 15 15 15 55
   :header-rows: 1

   * - Field
     - Type
     - Required
     - Description
   * - ``dist_id``
     - float
     - Yes
     - Target inter-planar separation distance in Å.
   * - ``dist_sd``
     - float
     - Yes
     - Standard deviation of the inter-planar distance in Å.
   * - ``angle_id``
     - float
     - No
     - Target angle between the two plane normals in degrees (default ``0.0``
       for parallel planes).
   * - ``angle_sd``
     - float
     - Yes
     - Standard deviation of the inter-planar angle in degrees.
   * - ``specs``
     - list
     - Yes
     - Two atom-group lists defining Plane 1 (``specs[0]``) and Plane 2
       (``specs[1]``), each resolving to three or more atoms.

.. note::
   Following Refmac's behaviour, only the inter-planar angle restraint is
   evaluated by default during refinement; ``dist_id`` and ``dist_sd`` are
   nonetheless required keys in the JSON definition. Note also that none of the
   ``scale_sigma_*`` defaults apply to ``stac``.

**Example:**

.. code-block:: json

   {
     "rest_type": "stac",
     "restr": {
       "dist_id": 3.4,
       "dist_sd": 0.2,
       "angle_id": 0.0,
       "angle_sd": 6.0,
       "specs": [
         [{"chain": "B", "resi": 50, "names": ["C2", "C4", "C5", "C6", "N1", "N3", "N4", "O2"]}],
         [{"chain": "B", "resi": 51, "names": ["C2", "C4", "C5", "C6", "C8", "N1", "N3", "N6", "N7", "N9"]}]
       ]
     }
   }

Harmonic restraints (``harm``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Applies a Refmac-style positional harmonic damping restraint to selected atoms
by adding :math:`1 / \sigma_t^2` to the diagonal of the positional
second-derivative matrix in each cycle. Rather than pulling atoms back toward
fixed initial coordinates, this damps positional shifts per cycle so restrained
atoms move more slowly than unrestrained atoms.

.. list-table::
   :widths: 15 15 15 55
   :header-rows: 1

   * - Field
     - Type
     - Required
     - Description
   * - ``rectype``
     - string
     - Yes
     - Selection record type, e.g. ``"atom"`` (explicit atom selection) or
       ``"resi"`` (residue-range selection). Must not be ``"auto"``.
   * - ``sigma_t``
     - float
     - Yes
     - Positional damping standard deviation in Å (smaller values restrict
       atomic shifts more strongly; not scaled by ``scale_sigma_*``).
   * - ``specs``
     - list
     - Yes
     - One or more single-element atom spec lists using either explicit atom
       selection or residue-range selection (``ifirst`` / ``ilast``).

**Examples:**

Restraining specific atoms:

.. code-block:: json

   {
     "rest_type": "harm",
     "restr": {
       "rectype": "atom",
       "sigma_t": 0.5,
       "specs": [
         [{"chain": "A", "resi": 100, "names": ["ZN"]}]
       ]
     }
   }

Restraining a residue range:

.. code-block:: json

   {
     "rest_type": "harm",
     "restr": {
       "rectype": "resi",
       "sigma_t": 0.5,
       "specs": [
         [{"chain": "A", "ifirst": 10, "ilast": 50, "atom": "*"}]
       ]
     }
   }

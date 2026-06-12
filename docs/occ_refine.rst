Occupancy refinement
====================

.. warning::
   **Experimental Feature:** Occupancy refinement in Servalcat is currently 
   experimental. The underlying algorithms, YAML syntax, and default behaviours 
   are subject to change in future releases.

   Currently, this requires low-level manual control via the configuration file. 
   Once stable recommendations and workflows are established, this process will 
   be automated so that users do not have to write these complex YAML files directly. 
   Use with caution.

Servalcat supports multiple occupancy refinement scenarios, including refining 
individual atom occupancies, grouped occupancies, and group-constrained occupancies. 
This configuration is managed via the refinement YAML input file.

.. note::
   Refinement with group constraints typically requires more cycles to reach convergence. 
   It is recommended to try 30 or more cycles and monitor the occupancy values 
   per cycle in the log file to ensure stability.
   
   It is also advisable to use the non-binary solvent mask when refining occupancies.
   Try the ``--non_binary_solvent_mask`` option.

Example configuration
---------------------

.. code-block:: yaml

   refine:
     atom_selection:
       occ:
         include: ["A/338/*"]
         exclude: []
     occ_groups:
       - id: 1
         selections: ["A/338/*:A"]
       - id: 2
         selections: ["A/338/*:B"]
     occ_group_constraints:
       - ids: [1, 2]
         complete: true
     occ_group_const_mu: 10
     occ_group_const_mu_update_factor: 1.1
     occ_group_const_mu_update_tol_rel: 0.25
     occ_group_const_mu_update_tol_abs: 0.01
     local_adpr_weights:
       - sel: "A/338/*"
         w: 5

Configuration options
---------------------

atom_selection.occ
~~~~~~~~~~~~~~~~~~
Defines which atoms are included or excluded from the overall occupancy refinement calculation. By default, no occupancies will be refined.

* **include**: A list of selection strings designating which atoms will have their occupancies refined (e.g., alternative conformations :A and :B of residue 338). See `Gemmi's documentation <https://gemmi.readthedocs.io/en/latest/analysis.html#selections-cid>`_ for the selection syntax.
* **exclude**: A list of selection strings defining specific atoms that should be excluded from the selections specified by ``include``. This parameter exists because the Gemmi selection syntax cannot always express complex exclusions natively.

occ_groups
~~~~~~~~~~
(Optional) Groups individual atom selections together to define a single, shared occupancy parameter.

* **id**: A unique integer identifying the occupancy group. This integer value is used to reference the group inside the ``occ_group_constraints`` block.
* **selections**: A list of selection strings defining the atoms that belong to this specific group ID.

.. note::
   If ``occ_groups`` is omitted from the configuration, individual occupancies for every atom matched in ``atom_selection.occ`` will be refined independently. It is usually undesirable and can destabilise the refinement.

occ_group_constraints
~~~~~~~~~~~~~~~~~~~~~
(Optional) Defines relationships and constraints between different occupancy groups. This is typically used to model mutually exclusive alternative conformations.

* **ids**: A list of group IDs (matching the ``id`` defined in ``occ_groups``) that are bound by this constraint.
* **complete**: A boolean that controls how the total occupancy of the specified groups is constrained:
  
  * When set to ``true``, it enforces a complete system where the sum of the occupancies of the listed group IDs must equal exactly 1.0 (i.e., :math:`\sum \text{occ} = 1.0`).
  * When set to ``false``, it enforces an incomplete system where the sum of the occupancies is constrained to be less than or equal to 1.0 (i.e., :math:`\sum \text{occ} \le 1.0`), which is useful if part of the conformation is disordered and unmodelled.

This setting also affects the setup of non-bonded (vdw) interactions in the geometry restraints. It determines which atom pairs should be accounted for; if the groups are mutually exclusive, they will not interact with each other.

local_adpr_weights
~~~~~~~~~~~~~~~~~~
(Optional) Configures localised adjustments to the atomic displacement parameter (ADP) restraint weights.

* **sel**: A selection string specifying the target atoms or residues.
* **w**: The weight multiplier applied to the ADP restraints for the selected atoms. The default value is 1.

.. tip::
   When refining occupancies, the occupancy and ADP values can easily become highly correlated, leading to unstable refinement or physically unrealistic B-factors. It is highly advisable to use a higher weight (tighter restraints) around the occupancy-refining atoms to stabilise the system.

Refinement behaviour summary
----------------------------
The presence or absence of the optional configuration sections changes how Servalcat behaves:

1. **Neither occ_groups nor occ_group_constraints defined:** Individual atom occupancies are refined independently.
2. **Only occ_groups defined:** All occupancies within the same group are constrained to be equal, and these grouped occupancies are refined without multi-group constraints.
3. **Both occ_groups and occ_group_constraints defined:** Grouped occupancies are refined, and the relationships specified in ``occ_group_constraints`` are enforced using the augmented Lagrangian method.

Understanding the augmented Lagrangian method
---------------------------------------------
When ``occ_group_constraints`` are defined, Servalcat utilises the augmented Lagrangian method to handle the mathematical boundaries (whether summing to exactly 1.0 or staying below 1.0).

For a target function :math:`f(x)` subject to constraint functions :math:`c_i(x) = 0`, the augmented Lagrangian function :math:`\mathcal{L}_A` is formulated as:

.. math::

   \mathcal{L}_A(x, \lambda, \mu) = f(x) - \sum_i \lambda_i c_i(x) + \sum_i \frac{\mu_i}{2} c_i(x)^2

Where:

* :math:`x` represents the atom or group occupancies being refined.
* :math:`f(x)` is the refinement target (negative log-likelihood).
* :math:`c_i(x) = 0` represents the occupancy constraint (e.g., :math:`\sum \text{occ} - 1.0 = 0` for ``complete: true``).
* :math:`\lambda_i` is the Lagrange multiplier for constraint :math:`i`.
* :math:`\mu_i` is the penalty parameter for constraint :math:`i`.

Update mechanism and parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
At the end of an iteration, the Lagrange multiplier :math:`\lambda_i` and the penalty parameter :math:`\mu_i` are dynamically updated based on the constraint values from the current cycle (:math:`c_i`) and the previous cycle (:math:`c_{i,\text{prev}}`):

.. math::

   \lambda_i \leftarrow \lambda_i - \mu_i c_i

.. math::

   \mu_i \leftarrow \begin{cases} \mu_i  & \text{if } |c_i| < \max(\epsilon, \eta |c_{i,\text{prev}}|) \\ \alpha \mu_i & \text{otherwise} \end{cases}

The parameters controlling this dynamic adjustment are described below:

* **occ_group_const_mu**: The initial penalty multiplier :math:`\mu` value for enforcing the occupancy constraints. Higher values penalise constraint violations more harshly from the start of the refinement.
* **occ_group_const_mu_update_factor**: The scaling factor :math:`\alpha`. If the constraint violation does not decrease sufficiently compared to the previous cycle, :math:`\mu` is multiplied by this factor to tighten control.
* **occ_group_const_mu_update_tol_rel**: The relative tolerance threshold :math:`\eta`. It determines whether the rate of constraint improvement is sufficient to keep the current penalty multiplier unchanged.
* **occ_group_const_mu_update_tol_abs**: The absolute tolerance threshold :math:`\epsilon`. If the absolute violation falls below this value, the constraint is considered satisfied and the penalty value is not increased.
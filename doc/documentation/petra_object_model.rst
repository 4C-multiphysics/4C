.. _petra-object-model:

Distributed computations with the Petra Object Model
====================================================

|FOURC| is heavily based on the Trilinos project.
For this reason, management of data distributed across a parallel machine
follows Trilinos' **Petra Object Model**.

Petra Object Model
------------------

The Petra Object Model is a foundational framework within the Trilinos project,
designed to facilitate the construction and manipulation of distributed memory objects in parallel computing environments.
It provides a structured approach to managing data distribution and communication across multiple MPI processes,
ensuring efficient parallel computations.

It uses the concept of a **map** for the

- **Distribution of global objects**: A map defines how a global object, such as a vector or matrix, is partitioned across the local memories of different MPI processes.
- **Global-to-Local Index Translation**: A map manages the translation between global indices (representing the entire dataset) and local indices (specific to a processes' subset), facilitating efficient data access and manipulation.

Types of Maps -- a linear algebra point of view
-----------------------------------------------

The Petra Object Model uses four maps: row map, column map, domain map, range map

Let's assume the matrix-vector product

.. math::
  y = Ax

with vectors :math:`x` and :math:`y` as well as the matrix :math:`A` being stored in a distributed memory environment.

- **Row Map**: The row map defines how the rows of a distributed matrix or the entries of a distributed vector are partitioned across multiple processes. It specifies which rows are owned by each process, facilitating efficient parallel operations. For instance, in a sparse matrix, the row map determines the distribution of matrix rows among processes.
- **Column Map**: The column map pertains to the columns of a distributed matrix. It indicates which columns are needed by each process to perform local computations. This is particularly important in sparse matrix-vector multiplications, where a process may require access to certain columns that are owned by other processes. The column map ensures that each process knows which columns to access and possibly import from other processes.
- **Domain Map**: The domain map defines the distribution of the input vector space for an operator or matrix. In the context of the exemplary matrix-vector product, the domain map describes how the elements of the vector :math:`x` are distributed across processes. This ensures that each process knows which portions of the input vector it is responsible for during the computation.
- **Range Map**: Conversely, the range map defines the distribution of the output vector space for an operator or matrix. In the context of the exemplary matrix-vector product, the range map describes how the elements of the resulting vector :math:`y`  are distributed across processes. This ensures that each process knows which portions of the output vector it is responsible for after the computation.

.. admonition:: Example

  Assume the matrix-vector multiplication :math:`y=Ax` to read

  .. math::
    \begin{bmatrix} y_1\\y_2\\y_3 \end{bmatrix}
    = \begin{bmatrix} 2 & -1 & 0\\-1 & 2 & -1\\0 & -1 & 2 \end{bmatrix}
    \begin{bmatrix} x_1\\x_2\\x_3 \end{bmatrix}.

  This will now be computed on two processes, *p0* and *p1*.
  Assume further that the first two rows are stored on *p0*, reading

  .. math::
    \begin{bmatrix} y_1\\y_2 \end{bmatrix}, \quad
    \begin{bmatrix} 2 & -1 & 0\\-1 & 2 & -1 \end{bmatrix}, \quad
    \begin{bmatrix} x_1\\x_2 \end{bmatrix},

  while the last row is stored on *p1*, reading

  .. math::
    \begin{bmatrix} y_3 \end{bmatrix}, \quad
    \begin{bmatrix} 0 & -1 & 2 \end{bmatrix}, \quad
    \begin{bmatrix} x_3 \end{bmatrix}.

  Then, the maps of the Petra Object Model read as follows:

  +-------------+---------+------+
  + Type of Map | *p0*    | *p1* |
  +=============+=========+======+
  | Row map     | 0, 1    | 2    |
  +-------------+---------+------+
  | Column map  | 0, 1, 2 | 1, 2 |
  +-------------+---------+------+
  | Domain map  | 0, 1    | 2    |
  +-------------+---------+------+
  | Range map   | 0, 1    | 2    |
  +-------------+---------+------+

  Note:

  - Rows are fully owned by a single MPI process.
  - Row map = Domain map = Range map (all maps are 1-to-1)
  - Column map is not 1-to-1

Types of Maps -- a finite element point of view
-----------------------------------------------

While domain map and range map are strongly tied to matrices,
the concept of row and column maps transfers directly to finite element discretizations,
where they are used to describe the distribution of mesh-based quantities (e.g., elements, nodes, or degrees of freedom)
across a parallel computer.

- **Row Map**: The row map specifies the mesh-based quantities (e.g., elements, nodes, degrees of freedoms) owned by each process.
- **Column Map**: The column indicates which mesh-based quantities (e.g., elements, nodes, degrees of freedom) are needed by each process to perform local computations. Off-process data must be acquired through communication between processes.

Surface areas and surface integrals
===================================

This chapter tries to explain the background of the ``ComputeMetricTensorForSurface`` function that can be found at various places in |FOURC|.

Motivation and definition.
''''''''''''''''''''''''''

(a) Let :math:`a^{1},\ldots,a^{k}\in\mathbb{R}^{n}`. Let\

.. math:: A:=\bigl(a^{1}\; a^{2}\;\cdots\; a^{k}\bigr)\in\mathbb{R}^{n\times k}

be the matrix with columns :math:`a^{1},\ldots,a^{k}`. Then\

.. math:: A\bigl([0,1]^{k}\bigr)=\bigl\{Ax;\,x\in\mathbb{R}^{k},\; x_{j}\in[0,1]\text{ for all }j=1,\ldots,k\bigr\}\subseteq\mathbb{R}^{n}

is the *parallelepiped* spanned by the vectors
:math:`a^{1},\ldots,a^{k}`. If :math:`k=2`, this is the *parallelogram*
spanned by the vectors :math:`a^1` and :math:`a^2`.

(b) In the case :math:`k=n`, it holds that
:math:`\operatorname{vol}_{n}\bigl(A\bigl([0,1]^{n}\bigr)\bigr)=\lvert\det A\rvert`.
If :math:`a^{1},\ldots,a^{n}` are linearly dependent, then both sides
are :math:`=0`, for :math:`A\bigl([0,1]^{k}\bigr)` has width :math:`0`
in (at least) one direction.

(c) The *Gramian matrix* of the vectors :math:`a^{1},\ldots,a^{k}` is
defined as

.. math::

   A^{\top}A = \bigl(\langle A^{\top}Ae_{i},e_{j}\rangle\bigr)_{i,j=1,\ldots,n}
               = \bigl(\langle Ae_{i},Ae_{j}\rangle\bigr)_{i,j=1,\ldots,n}
   	    = \bigl(\langle a_{i},a_{j}\rangle\bigr)_{i,j=1,\ldots,n}
               \in\mathbb{R}^{n\times n}

(with :math:`\langle\,\cdot\,,\,\cdot\,\rangle` being the standard
scalar product in :math:`\mathbb{R}^{k}` and :math:`e_{i}` the
:math:`i`\ th standard basis vector of :math:`\mathbb{R}^{k}`). It is
positive-semidefinite (because
:math:`\langle A^{\top}Ax,x\rangle=\lvert Ax\rvert^{2}\ge0` for all
:math:`x\in\mathbb{R}^{k}`) and therefore :math:`\det A^{\top}A\ge0`.

Now we define\

.. math:: \gamma(A):=\sqrt{\det(A^{\top}A)}=\sqrt{\det\bigl(\langle Ae_{i},Ae_{j}\rangle\bigr)}.

Notice that :math:`\gamma(A)=\lvert\det A\rvert` if :math:`k = n`.

(d) We want to motivate why :math:`\gamma(A)` is the
:math:`k`-dimensional volume of :math:`A\bigl([0,1]^{k}\bigr)`. Here is
an example: Let :math:`n=k+1` and
:math:`A\bigl([0,1]^{k}\bigr)\subseteq\mathbb{R}^{k}\times\{0\}`, e.g.
the parallelepiped has width :math:`0` in the direction of
:math:`e_{n}=e_{k+1}`. Let :math:`Q:\mathbb{R}^{k+1}\to\mathbb{R}^{k}`
be the projection onto the first :math:`k` coordinates (thus
:math:`Q(x)=Q\bigl((x_{1},\ldots,x_{n})\bigr)=(x_{1},\ldots,x_{k})`),
then\

.. math:: (QA)^{\top}QA=A^{\top}Q^{\top}QA=A^{\top}A,

and therefore
:math:`\operatorname{vol}_{k}(QA\bigl([0,1]^{k}\bigr))=\lvert\det(QA)\rvert=\gamma(A)`,
using part (b). So in this case, :math:`\gamma(A)` is indeed the
:math:`k`-dimensional volume of :math:`A\bigl([0,1]^{k}\bigr)`.

Integration on submanifolds.
''''''''''''''''''''''''''''

Now let :math:`M\subseteq\mathbb{R}^{n}` be a :math:`k`-dimensional
submanifold of :math:`\mathbb{R}^{n}`, with global parameterization.
While we won’t give the exact definition of submanifolds here, this
basically means that there is some open set
:math:`\Omega\subseteq\mathbb{R}^{k}` and a parameterization
:math:`\Phi:\Omega\to M` that is, among other things, smooth and
one-to-one. Also, let :math:`f:M\to\mathbb{R}` be a suitable function
(again, we don’t give the exact requirements here). Then we define the
*surface integral*\

.. math:: \int_{M}f(x)\,\mathrm{d}S(x) := \int_{\Omega}f(\Phi(\xi))\,\gamma(\Phi'(\xi))\,\mathrm{d}\xi.

In particular, we define\

.. math:: \operatorname{vol}_{k}(M):=\int_{M}1\,\mathrm{d}S(x) =\int_{\Omega}\gamma(\Phi'(\xi))\,\mathrm{d}\xi.

Example.
''''''''

A parameterization  [1]_ of the two-dimensional unit sphere
:math:`S_{2}:=\{x\in\mathbb{R}^{3};\,\lvert x\rvert=1\}` in
:math:`\mathbb{R}^{3}` is\

.. math::

   \begin{gathered}
   \Phi:(-\pi,\pi)\times(-\tfrac{\pi}{2},\tfrac{\pi}{2})\to\mathbb{R}^{3},\\
   \Phi(\varphi_{1},\varphi_{2}):=\begin{pmatrix}\cos\varphi_{1}\cos\varphi_{2}\\
   \sin\varphi_{1}\cos\varphi_{2}\\
   \sin\varphi_{2}\end{pmatrix}.
   \end{gathered}

Its derivative (the Jacobian matrix) is\

.. math::

   \Phi'(\varphi_{1},\varphi_{2})=\begin{pmatrix}-\sin\varphi_{1}\cos\varphi_{2} & -\cos\varphi_{1}\sin\varphi_{2}\\
   \cos\varphi_{1}\cos\varphi_{2} & -\sin\varphi_{1}\sin\varphi_{2}\\
   0 & \cos\varphi_{2}\end{pmatrix},

and thus\

.. math::

   \Phi'(\varphi_{1},\varphi_{2})^{\top}\Phi'(\varphi_{1},\varphi_{2})=\begin{pmatrix}\cos^{2}\varphi_{2} & 0\\
   0 & 1\end{pmatrix},

so we see that
:math:`\gamma(\Phi'(\varphi_{1},\varphi_{2}))=\cos\varphi_{2}`. Now we
compute the 2-dimensional volume, i.e. the surface area of the unit
sphere, by\

.. math:: \operatorname{vol}_{2}(S_{2})=\int_{S_{2}}1\,\mathrm{d}S(x)=\int_{\varphi_{1}=-\pi}^{\pi}\int_{\varphi_{2}=-\pi/2}^{\pi/2}\cos\varphi_{2}\,\mathrm{d}\varphi_{2}\,\mathrm{d}\varphi_{1}=\left.2\pi\sin\varphi_{2}\right|_{-\pi/2}^{\pi/2}=4\pi.

Remark.
'''''''

For vectors :math:`a,b\in\mathbb{R}^{3}` in the three-dimensional space
:math:`\mathbb{R}^{3}` it holds that\

.. math::

   \lvert a\times b\rvert^{2}=\langle a\times b,a\times b\rangle=\langle a,a\rangle\langle b,b\rangle-\langle a,b\rangle^{2}=\det\begin{pmatrix}\langle a,a\rangle & \langle a,b\rangle\\
   \langle b,a\rangle & \langle b,b\rangle\end{pmatrix},

and thus if :math:`\Phi(t)=\Phi(t_{1},t_{2})`, then\

.. math:: \lvert\partial_{t_{1}}\Phi\times\partial_{t_{2}}\Phi\rvert^{2}=\det\bigl(\langle\partial_{t_{i}}\Phi,\partial_{t_{j}}\Phi\rangle_{i,j}\bigr)=\det(\Phi'^{\top}\Phi').

The matrix :math:`\Phi'^{\top}\Phi'`, i.e. the Gramian matrix of the
vectors :math:`\partial_{t_{1}}\Phi` and :math:`\partial_{t_{2}}\Phi`,
is also called the *metric tensor*, and the expression

.. math:: \lvert\partial_{t_{1}}\Phi\times\partial_{t_{2}}\Phi\rvert\,\mathrm{d}t_{1}\,\mathrm{d}t_{2}=\gamma(\Phi')\,\mathrm{d}S

is known in engineering as the *area element.*

.. [1]
   Actually, this is not a parameterization of all of the unit sphere:
   The half-plane :math:`\{x\in\mathbb{R}^{3};\,x_{2}=0,x_{1}\le 0\}` is
   missing. A global parameterization of the unit sphere doesn’t exist,
   and the missing half-plane is a set of :math:`2`-dimensional measure
   zero, so we can disregard it.

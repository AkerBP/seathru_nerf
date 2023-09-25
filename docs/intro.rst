.. _intro-label:

Introduction
============

With Neural Radiance Fields (NeRFs), we can store a 3D scene as a continuous function.
This idea was first introduced in the original NeRF publication :cite:`nerf`.
Since then, the field experienced many advancements. Some of them even in the subsea domain.
However, these advancements still have some limitations. This implementation adresses some of those limitations
and provides a modular and documented implementation of a subsea specific NeRF that allows for easy modifications and experiments.

Approach
********
The fundamental principle underlying NeRFs is to represent a scene as a continuous function that maps a position,
:math:`\mathbf{x} \in \mathbb{R}^{3}`, and a viewing direction, :math:`\boldsymbol{\theta} \in \mathbb{R}^{2}`,
to a color :math:`\mathbf{c} \in \mathbb{R}^{3}` and volume density :math:`\sigma`. We can approximate this
continuous scene representation with a simple Multi Layer Perceptron (MLP).
:math:`F_{\mathrm{\Theta}} : (\mathbf{x}, \boldsymbol{\theta}) \to (\mathbf{c},\sigma)`.

It is common to also use positional and directional encodings to improve the performance of NeRF approaches. Furthermore,
there are various approaches in order to sample points in regions of a scene that are relevant to the final image. A detailed
explanation of the exact implemented architecture is given in the :ref:`architecture-label` section.

Image formation model
---------------------
The authors of :cite:`seathru_nerf` combine the fundamentals of NeRFs with the following underwater image formation model
proposed in :cite:`seathru`:

.. math::

   I = \overbrace{\underbrace{J}_{\text{colour}} \cdot \underbrace{(e^{-\beta^D(\mathbf{v}_D)\cdot z})}_{\text{attenuation}}}^{\text{direct}} + \overbrace{\underbrace{B^\infty}_{\text{colour}} \cdot \underbrace{(1 - e^{-\beta^B(\mathbf{v}_B)\cdot z})}_{\text{attenuation}}}^{\text{backscatter}}

:math:`I` ............... Image

:math:`J` ............... Clear image (without any water effects like attenuation or backscatter)

:math:`B^\infty` ........... Backscatter water colour at depth infinity

:math:`\beta^D(\mathbf{v}_D)` ... Attenuation coefficient [#f1]_

:math:`\beta^B(\mathbf{v}_B)` ... Backscatter coefficient [#f1]_

:math:`z` ............... Camera range

This image formation model allows the model to seperate between the clean scene and the water effects. This is very useful
since it allows for filtering out of water effects from a scene. Some results where this was achieved are shown in the
:ref:`results-label` section.


Rendering equations
-------------------
As NeRFs require a discrete and differentiable volumetric rendering equation, the authors of :cite:`seathru_nerf` propose
the following formulation:

.. math::

    \hat{\boldsymbol{C}}(\mathbf{r}) = \sum_{i=1}^N \hat{\boldsymbol{C}}^{\text{obj}}_i(\mathbf{r}) + \sum_{i=1}^N \hat{\boldsymbol{C}}^{\text{med}}_i(\mathbf{r})

This equation features an object and a medium part contributing towards the final rendered pixel
colour :math:`\hat{\boldsymbol{C}}(\mathbf{r})`. Those two components are given by:

.. math::

    \hat{\boldsymbol{C}}^{\text{obj}}_i(\mathbf{r}) = T^{\text{obj}}_i \cdot \exp (-\boldsymbol{\sigma}^{\text{attn}} t_i) \cdot \left(1 - \exp({-\sigma^{\text{obj}}_i \delta_i})\right) \cdot \mathbf{c}^{\text{obj}}_i

.. math::

    \hat{\boldsymbol{C}}^{\text{med}}_i(\mathbf{r}) = T^{\text{obj}}_i \cdot \exp (-\boldsymbol{\sigma}^{\text{bs}} t_i) \cdot \left(1 - \exp({-\boldsymbol{\sigma}^{\text{bs}} \delta_i})\right) \cdot \mathbf{c}^{\text{med}}

, with

.. math::

    T^{\text{obj}}_i = \exp\left(-\sum_{j=0}^{i-1}\sigma^{\text{obj}}_j\delta_j\right)

The above equations contain five parameters that are used to describe the underlying scene:
object density :math:`\sigma^{\text{obj}}_i \in \mathbb{R}^{1}`, object colour
:math:`\mathbf{c}^{\text{obj}}_i \in \mathbb{R}^{3}`, backscatter density
:math:`\boldsymbol{\sigma}^{\text{bs}} \in \mathbb{R}^{3}`, attenuation density
:math:`\boldsymbol{\sigma}^{\text{attn}} \in \mathbb{R}^{3}`, and medium colour
:math:`\mathbf{c}^{\text{med}} \in \mathbb{R}^{3}`.

I use the network discussed below to compute those five parameters that parametrize the underlying scene.

.. _architecture-label:

Network architecture
--------------------

The network implemented for this approach has the following architecture:

.. image:: media/my_architecture.png
   :align: center
   :alt: Network architecture

.. raw:: html

    <br>

The object network computes :math:`\sigma^{\text{obj}}_i` and :math:`\mathbf{c}^{\text{obj}}_i`, while the
medium network computes :math:`\boldsymbol{\sigma}^{\text{bs}}`, :math:`\boldsymbol{\sigma}^{\text{attn}}` and
:math:`\mathbf{c}^{\text{med}}`.

The proposal network is used to sample point in regions of the scene that contribute most to the final image. This approach
actually uses two proposal networks that are connected sequentially. More details on the concept of proposal samplers and
how they are optimized during training can be found in :cite:`mipnerf360`.

For positional encoding, I use Hash Grid Encodings as proposed in :cite:`instant-ngp` and for directional
encoding I use Spherical Harmonics Encoding (SHE) introduced in :cite:`refnerf`.

The MLPs in the object and medium networks are implemented using `tinycuda-nn <https://github.com/NVlabs/tiny-cuda-nn>`_ for
performance reasons.

.. rubric:: Footnotes

.. [#f1] Those depend on range, object reflectance, spectrum of ambient light, the camera's spectral response, and the physical scattering and beam attenuation coefficients of the water, all of which are wavelength-dependent.


.. rubric:: References

.. bibliography:: references.bib
    :style: plain
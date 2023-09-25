.. _installation-label:

Installation
============

This repository is built upon the `nerfstudio <https://github.com/nerfstudio-project/nerfstudio/>`_
library. This library has a few requirements, with the most important being having access
to a CUDA compatible GPU. Furthermore, for full functionality, it needs `colmap <https://colmap.github.io/>`_,
`ffmpeg <https://ffmpeg.org/>`_ and `tinycuda-nn <https://github.com/NVlabs/tiny-cuda-nn>`_.
Those requirements might be a bit tricky to install and the installation is system dependent.
The easiest way to install them is to follow the instructions `here <https://docs.nerf.studio/en/latest/quickstart/installation.html>`__
up to the point of actually installing nerfstudio.

For this approach, I extended the nerfstudio library. Therefore please install my fork of
nerfstudio available `here <https://github.com/acse-pms122/nerfstudio_dev>`__. This can be
done via:

.. code-block:: bash

    git clone https://github.com/acse-pms122/nerfstudio_dev.git
    cd nerfstudio_dev
    pip install -e .

Then, clone this repository and install it via:

.. code-block:: bash

    cd ..
    git clone https://github.com/ese-msc-2022/irp-pms122.git
    cd irp-pms122
    pip install -e .

Then, install the command line completion via:

.. code-block:: bash

    ns-install-cli

To check the installation, type:

.. code-block:: bash

    ns-train seathru-nerf --help

If you see the help message, you are good to go! 🚀🚀🚀


Requirements
************

This implementation requires a GPU with a CUDA copatible driver. There are two model configurations as summarised
in the following table:

.. list-table::
   :header-rows: 1
   :widths: 20 40 10 10

   * - Method
     - Description
     - Memory
     - Quality
   * - ``seathru-nerf``
     - Larger model, used to produced results in report
     - ~23 GB
     - Best
   * - ``seathru-nerf-lite``
     - Smaller model
     - ~7 GB
     - Good

I recommend to use the ``seathru-nerf`` method as it was used to experiment and produce the results presented in the paper.
The ``seathru-nerf-lite`` still produces good results, but has not been tested on all scenes. If you happen to run into a
``CUDA_OUT_MEMORY_ERROR`` it is a sign that the available VRAM on the GPU is not enough. You can either use the smaller
model, decrease the batch size, do both or upgrade to a better GPU.
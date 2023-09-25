.. _results-label:

Results
=======

This section shows off some results that can be achieved with this repository. To get a sense of the capabilities and
limitations of the implemented approach, have a look at some of them!🔍

Renderings
**********

This are some results that show the reconstructed scene on the left and the clean scene on the right.


.. raw:: html

    <details>
        <summary>Click to see renderings of IUI3-RedSea</summary>
        <iframe width="560" height="315" src="https://www.youtube.com/embed/EjNc-o3LptY?si=wXmoviLHrwdfTHDc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </details>


.. raw:: html

    <details>
        <summary>Click to see renderings of Curasao</summary>
        <iframe width="560" height="315" src="https://www.youtube.com/embed/PO6vF60tdDQ?si=C0c-VQ_r6nf53EBn" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </details>

.. raw:: html

    <details>
        <summary>Click to see renderings of JapaneseGardens-RedSea</summary>
        <iframe width="560" height="315" src="https://www.youtube.com/embed/EMFzwOsSNKQ?si=SvmtJUpADdgQFSGR" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </details>

.. raw:: html

    <br>

The implementation can also be used to render depthmaps and object weight accumulation maps of the scene. The following videos
show the depthmaps on the left and the object weight accumulation maps on the right. From the object accumulation maps, we
can nicely see, that the model is able to seperate between objects (red hue, indicating high accumulation) and the water (blue
hue, indicating low accumulation).

.. raw:: html

    <details>
        <summary>Click to see renderings of IUI3-RedSea</summary>
        <iframe width="560" height="315" src="https://www.youtube.com/embed/UKqaSy9rJnA?si=3v9WdqxpmvvpEBPO" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </details>

.. raw:: html

    <details>
        <summary>Click to see renderings of Curasao</summary>
        <iframe width="560" height="315" src="https://www.youtube.com/embed/qEBe0dL4kMc?si=3cnfkPgyKBBgF13Y" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </details>

.. raw:: html

    <details>
        <summary>Click to see renderings of JapaneseGardens-RedSea</summary>
        <iframe width="560" height="315" src="https://www.youtube.com/embed/YW33VPZYGg0?si=xKS55gbN4W8rByQ0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
    </details>

.. raw:: html

    <br>

The model also allows to render only the backscatter of the scene and the clear scene but with attenuation effects.

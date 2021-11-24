.. _sec-discopy:

DisCoPy
=======

While the :ref:`parser <sec-parsing>` provides ``lambeq``'s input, *DisCoPy* [#f1]_ [FTC2020]_ is ``lambeq``'s underlying engine, the component where all the low-level processing takes place. At its core, DisCoPy is a Python library that allows computation with :term:`monoidal categories <monoidal category>`. The main data structure is that of a *monoidal diagram*, or :ref:`string diagram <sec-string-diagrams>`, which is the format that ``lambeq`` uses internally to encode a sentence (:py:class:`discopy.rigid.Diagram`). DisCoPy makes this easy, by offering many language-related features, such as support for :term:`pregroup grammars <pregroup grammar>` and :term:`functors <functor>` for implementing :term:`compositional models <compositional model>` such as :term:`DisCoCat`. Furthermore, from a quantum computing perspective, DisCoPy provides abstractions for creating all standard :term:`quantum gates <quantum gate>` and building :term:`quantum circuits <quantum circuit>`, which are used by ``lambeq`` in the final stages of the :ref:`pipeline <sec-pipeline>`.

Thus, it is not a surprise that the advanced use of ``lambeq``, involving extending the toolkit with new :term:`compositional models <compositional model>` and :term:`ansätze <ansatz (plural: ansätze)>`, requires some familiarity of DisCoPy. For this, you can use the following resources:

- For a gentle introduction to basic DisCoPy concepts, start with ``lambeq``'s tutorial :ref:`sec-advanced`.
- The `basic example notebooks <https://discopy.readthedocs.io/en/main/notebooks.basics.html>`_ in DisCoPy documentation provide another good starting point.
- The `advanced tutorials <https://discopy.readthedocs.io/en/main/notebooks.advanced.html>`_ in DisCoPy documentation can help you to delve further into DisCoPy.

.. rubric:: Footnotes

.. [#f1] https://github.com/oxford-quantum-group/discopy

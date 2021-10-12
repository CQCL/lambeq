.. sec-discopy:

DisCoPy
=======

While the :ref:`parser <sec-parsing>` provides ``lambeq``'s input, *DisCoPy* [1]_ is ``lambeq``'s underlying engine, the component where all the low-level processing takes place. At its core, DisCoPy is a Python library that allows computation with monoidal categories. The main data structure is that of a *monoidal diagram*, or :ref:`string diagram <sec-string-diagrams>`, which is the format that ``lambeq`` uses internally to encode a sentence (:py:class:`discopy.rigid.Diagram`). DisCoPy makes this easy, by offering many language-related features, such as support for pregroup grammars and functors for implementing compositional models such as DisCoCat. Furthermore, from a quantum computing perspective, DisCoPy provides abstractions for creating all standard quantum gates and building quantum circuits, which are used by ``lambeq`` in the final stages of the :ref:`pipeline <sec-pipeline>`.

Thus, it is not a surprise that advanced use of ``lambeq``, involving extending the toolkit with new compositional models and ansätze, requires some familiarity of DisCoPy. The `basic example notebooks <https://discopy.readthedocs.io/en/main/notebooks.basics.html>`_ provide a gentle introduction to basic DisCoPy concepts.

.. [1] https://github.com/oxford-quantum-group/discopy

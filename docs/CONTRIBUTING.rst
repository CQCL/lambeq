.. _sec-contributing:

Contributing to lambeq
======================

Contributions to ``lambeq`` are welcome, especially with regard to adding:

- Support for new :term:`parsers <parser>` (extensions of the :py:class:`.CCGParser` class)
- :term:`Compositional schemes <compositional model>` and :term:`readers <reader>` (extensions of the :py:class:`.Reader` class)
- :term:`Rewrite rules <rewrite rule>` (extensions of the :py:class:`.RewriteRule` class)
- Tensor and circuit :term:`ansätze <ansatz (plural: ansätze)>` (extensions of the :py:class:`.TensorAnsatz` and :py:class:`.CircuitAnsatz` classes)
- New :term:`trainers <trainer>`, :term:`models <model>`, and optimizers for the :py:mod:`.training` package.

All accepted contributions will be included in the next official release and contributors will be properly attributed in the corresponding release notes.

Opening a pull request
----------------------

If you have an already implemented and tested proposal, you can `open a pull request <https://github.com/CQCL/lambeq/pulls>`_ that will be reviewed by ``lambeq``'s development team. Keep in mind the following guidelines:

- Please provide a detailed description of your proposal, supporting it with references to publications or other material when appropriate. Suggestions for untested or ad-hoc components whose motivation is not well-defined cannot be accepted. If you are not sure about your idea, it would be preferable to contact the development team and discuss it or :ref:`open an issue <open-issue>` before opening a pull request.

- Examine the `existing code <https://github.com/CQCL/lambeq/tree/main/lambeq>`_ and try to apply the same conventions for styling, formatting, and documenting in your pull request. In general, we try to follow the standard `PEP-8 Python Style Guide <https://www.python.org/dev/peps/pep-0008/>`_ - if you are not familiar with it please have a look before opening a pull request. Docstrings use the `numpydoc conventions <https://numpydoc.readthedocs.io/en/latest/>`_.

- The signatures of all methods (public or private) need to be `type-annotated`. Please refer to the `Python typing module <https://docs.python.org/3/library/typing.html>`_ for more information.

- Try to accompany any proposed new functionality with a set of appropriate tests. The test coverage of ``lambeq`` is close to 100% and we would like to keep it that way. Please have a look at the `existing tests <https://github.com/CQCL/lambeq/tree/main/tests>`_ to get an idea about the conventions we use, or contact the dev team for guidance.

Trivial contributions
---------------------

Any contribution, no matter how small or "trivial", is welcome as long as it improves the package in a pragmatic and clear way. However, it is up to the maintainers of the project to decide if the sole purpose of a contribution is to add the author's name to the list of contributors, without providing any actual value to the development. We regret that these cases will not be accepted. Examples include the following:

- Changing the name of a variable without apparent reason.
- Rephrasing a comment without apparent reason.
- Adding an unnecessary comment.

As mentioned above, any contribution that genuinely improves the state of the code, no matter how small or "trivial", is welcome. For example:

- Fixing a small typo in a comment.
- Adding a type annotation that is missing.
- A minor formatting fix to improve compliance with `PEP-8 Python Style Guide <https://www.python.org/dev/peps/pep-0008/>`_.

.. _open-issue:

Opening an issue
----------------

If you have a question, proposal, or request related to ``lambeq``, please `open an issue <https://github.com/CQCL/lambeq/issues>`_ or send an email to lambeq-support@cambridgequantum.com. Keep an eye on the issues you have opened, and be sure to answer any questions from the developers to help them understand better the case. Issues that remain inactive for more than a week without an apparent reason will be marked as stale and eventually will be closed.

Where to start
--------------

For developers who wish to contribute to ``lambeq``, a good starting point would be the :ref:`UML diagrams <uml-diagrams>` provided for each sub-package, which give a high-level overview of their general structure as well as information regading the important external dependencies. General information for each ``lambeq`` sub-package can be also found in :ref:`this page <sec-package-api>`.

Code of conduct
---------------

Please be polite and respectful in any form of communication you have with other contributors/developers. Project maintainers are expected to take appropriate and fair corrective action in response to any instances of unacceptable behaviour. Project maintainers have the right and responsibility to remove, edit, or reject comments, commits, code, wiki edits, issues, and other contributions that are not aligned to these guidelines, or to ban temporarily or permanently any contributor for other behaviours that they deem inappropriate, threatening, offensive, or harmful.

.. rubric:: See also:

- :ref:`General information about sub-packages <sec-package-api>`
- :ref:`UML diagrams for sub-packages <uml-diagrams>`
- `"Extending lambeq" tutorial <tutorials/extend-lambeq.ipynb>`_
- `"DisCoPy usage" tutorial <advanced.rst>`_

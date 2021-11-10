.. _sec-contributing:

Contributing to lambeq
======================

Contributions to ``lambeq`` are welcome, especially with regard to adding:

- Support for new parsers (extensions of the :py:class:`.CCGParser` class)
- Compositional schemes/readers (extensions of the :py:class:`.Reader` class)
- Rewrite rules (extensions of the :py:class:`.RewriteRule` class)
- Tensor and circuit ansätze (extensions of the :py:class:`.TensorAnsatz` and :py:class:`.CircuitAnsatz` classes)

All accepted contributions will be included in the next official release and contributors will be properly attributed in the corresponding release notes.

Opening a pull request
----------------------

If you have an already implemented and tested proposal, you can `open a pull request <https://github.com/CQCL/lambeq/pulls>`_ that will be reviewed by ``lambeq``'s development team. Keep in mind the following guidelines:

- Please provide a detailed description of your proposal, supporting it with references to publications or other material when appropriate. Suggestions for untested or ad-hoc components whose motivation is not well-defined cannot be accepted. If you are not sure about your idea, it would be preferable to contact the development team and discuss it or :ref:`open an issue <open-issue>` before opening a pull request.

- Examine the `existing code <https://github.com/CQCL/lambeq/tree/main/lambeq>`_ and try to apply the same conventions for styling, formatting, and documenting in your pull request. In general, we try to follow the standard `PEP-8 Python Style Guide <https://www.python.org/dev/peps/pep-0008/>`_ - if you are not familiar with it please have a look before opening a pull request. Docstrings use the `numpydoc conventions <https://numpydoc.readthedocs.io/en/latest/>`_.

- The signatures of all methods (public or private) need to be `type-annotated`. Please refer to the `Python typing module <https://docs.python.org/3/library/typing.html>`_ for more information.

- Try to accompany any proposed new functionality with a set of appropriate tests. The test coverage of ``lambeq`` is close to 100% and we would like to keep it that way. Please have a look at the `existing tests <https://github.com/CQCL/lambeq/tree/main/tests>`_ to get an idea about the conventions we use, or contact the dev team for guidance.

.. _open-issue:

Opening an issue
----------------

If you have a question, proposal, or request related to ``lambeq``, please `open an issue <https://github.com/CQCL/lambeq/issues>`_ or send an email to lambeq-support@cambridgequantum.com. Keep an eye on the issues you have opened, and be sure to answer any questions from the developers to help them understand better the case. Issues that remain inactive for more than a week without an apparent reason will be marked as stale and eventually will be closed.

Code of conduct
---------------

Please be polite and respectful in any form of communication you have with other contributors/developers. Project maintainers are expected to take appropriate and fair corrective action in response to any instances of unacceptable behavior. Project maintainers have the right and responsibility to remove, edit, or reject comments, commits, code, wiki edits, issues, and other contributions that are not aligned to these guidelines, or to ban temporarily or permanently any contributor for other behaviors that they deem inappropriate, threatening, offensive, or harmful.

**See also:**

- `"Extending lambeq" tutorial <tutorials/extend-lambeq.ipynb>`_
- `"DisCoPy usage" tutorial <advanced.rst>`_

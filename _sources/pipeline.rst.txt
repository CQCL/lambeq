.. _sec-pipeline:

Pipeline
========

In ``lambeq``, the conversion of a sentence into a quantum circuit goes through the steps shown in :numref:`fig-pipeline`.

.. _fig-pipeline:
.. figure:: ./_static/images/pipeline.png 

   The general pipeline.

In more detail:

1. A syntax tree for the sentence is obtained by calling a statistical :ref:`CCG parser <sec-parsing>`. ``lambeq`` is equipped with a detailed API that greatly simplifies this process, and is shipped with support for a state-of-the-art parser.
 
2. Internally, the parse tree is converted into a :ref:`string diagram <sec-string-diagrams>`. This is an abstract representation of the sentence reflecting the relationships between the words as defined by the compositional model of choice, independently of any implementation decisions that take place at a lower level.

3. The string diagram can be simplified or otherwise transformed by the application of :ref:`rewriting rules <sec-rewrite>`; these can be used for example to remove specific interactions between words that might be considered redundant for the task at hand, or in order to make the computation more amenable to implementation on a quantum processing unit.
 
4. Finally, the resulting string diagram can be converted into a concrete *quantum circuit* (or a *tensor network* in the case of a "classical" experiment), based on a specific :ref:`parameterisation <sec-parameterise>` scheme and concrete choices of ansätze. ``lambeq`` features an extensible class hierarchy containing a selection of pre-defined ansätze, appropriate for both classical and quantum experiments.

After the last step, the output of the pipeline (quantum circuit or tensor network) is ready to be used for training. In the case of a fully quantum pipeline, the quantum circuit will be processed by a quantum compiler and subsequently uploaded onto a quantum computer, while in the classical case the tensor network will be passed to an ML or optimisation library, such as PyTorch or JAX. This first version of ``lambeq`` does not include any optimisation or training features of its own.

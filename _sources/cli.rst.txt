.. highlight:: bash

.. _sec-cli:

Command-line interface
======================

While ``lambeq`` is primarily aimed for programmatic use, since Release :ref:`rel-0.2.0` it is also equipped with a command-line interface that provides immediate and easy access to most of the toolkit's functionality. For example, this addition allows ``lambeq`` to be used as a dual :term:`parser`, capable of providing syntactic derivations in both :term:`pregroup <pregroup grammar>` and :term:`CCG <Combinatory Categorial Grammar (CCG)>` form.

A summary of the available options is given below.

::

    lambeq [-h] [-v] [-m {string-diagram,ccg}] [-i INPUT_FILE]
           [-f {json,pickle,text-unicode,text-ascii,image}]
           [-g {png,pdf,jpeg,jpg,eps,pgf,ps,raw,rgba,svg,svgz,tif,tiff}]
           [-u [KEY=VAR ...]] [-o OUTPUT_FILE | -d OUTPUT_DIR]
           [-p {bobcat,depccg}] [-t] [-s] [-r {spiders,stairs,cups,tree}]
           [-c [ROOT_CAT ...]] [-w [REWRITE_RULE ...]]
           [-a {iqp,tensor,spider,mps}] [-n [KEY=VAR ...]] [-y STORE_ARGS]
           [-l LOAD_ARGS]
           [input_sentence]

To get detailed help about the available options, type:

.. code-block:: console

    $ lambeq --help

The following sections provide an introduction to the command-line interface usage via specific examples, while all available options are described in depth in Section :ref:`Detailed Options <sec-detailed_options>`.

.. _sec-basic_usage:

Basic usage
-----------

The most straightforward use of the command-line interface of ``lambeq`` is to use it as a :term:`pregroup <pregroup grammar>` or :term:`CCG <Combinatory Categorial Grammar (CCG)>` :term:`parser`. The output formalism is controlled by the ``--mode`` option, which can be set to ``string-diagram`` or ``ccg``, with the first option to be the default. 

For example, to get the pregroup diagram for a sentence, just use the following command:

.. code-block:: console

    $ lambeq "John gave Mary a flower"

    John       gave      Mary    a    flower
    ────  ─────────────  ────  ─────  ──────
     n    n.r·s·n.l·n.l   n    n·n.l    n
     ╰─────╯  │  │   ╰────╯    │  ╰─────╯
              │  ╰─────────────╯

``lambeq`` will use the default :py:class:`~lambeq.BobcatParser` to parse the sentence and output the string diagram in the console with text drawing characters.

In order to get the corresponding CCG derivation, type:

.. code-block:: console

    $ lambeq -m ccg "John gave Mary a flower"

    John     gave      Mary   a   flower                                                                                
    ════  ═══════════  ════  ═══  ══════
     n    ((s\n)/n)/n   n    n/n    n   
          ────────────────>  ──────────>
              (s\n)/n            n     
          ─────────────────────────────>
                      s\n              
    ───────────────────────────────────<
                      s                  

Use the following command to read an entire file of sentences, tokenise them, parse them with the default parser, and store the pregroup or CCG diagrams in a new file:

.. code-block:: console

    $ lambeq -i sentences.txt -t -o diagrams.txt

.. note::
    For the rest of this document, all examples use the default string-diagram mode.

In the above example, file ``sentences.txt`` is expected to contain one sentence per line. The output will be written to file ``diagrams.txt``.
In case your input file does not contain one sentence per line, you can add the ``--split_sentences`` or ``-s`` flag.

If the text output is not good enough for your purposes, you can ask ``lambeq`` to prepare images for the diagrams in a variety of formats and store them in a specific folder for you:

.. code-block:: console

    $ lambeq -i sentences.txt -t -d image_folder -f image -g png

``lambeq`` will prepare a ``png`` file for each one of the sentences, and store it in folder ``image_folder`` using the line number of the sentence in the input file to name the image file, e.g. ``diagram_1.png``, ``diagram_2.png`` and so on.

.. note::
    Image generation is currently available only in string-diagram mode.
    
It is also possible to parse a single sentence and store it as an image -- for example, in PDF format in order to use it in a paper. In this case, you can name the file yourself and apply specific format options, such as the exact size of the figure or the font size used in the diagram. Note that it is not necessary to specify the image format if it is already contained in the file name (e.g. pdf).

.. code-block:: console

    $ lambeq -f image -u fig_width=16 fig_height=3 fontsize=12
    >        -o diagram.pdf
    >        "Mary does not like John"

.. _sec-advanced_options:

Using a reader
--------------

.. Note::
    Option only applicable to string diagrams.

Instead of the parser, users may prefer to apply one of the available :term:`readers <reader>`, each corresponding to a different :term:`compositional scheme <compositional model>`. For example, to encode a sentence as a :term:`tensor train`:

.. code-block:: console

    $ lambeq -r cups "John gave Mary a flower"

    START   John   gave   Mary    a    flower
    ─────  ─────  ─────  ─────  ─────  ──────
      s    s.r·s  s.r·s  s.r·s  s.r·s  s.r·s
      ╰─────╯  ╰───╯  ╰───╯  ╰───╯  ╰───╯  │

Readers can be used for batch processing of entire files with the ``-i`` option, exactly as in the parser case.

.. code-block:: console

    $ lambeq -r cups -i sentences.txt -o diagrams.txt

.. note::
    Some readers, such as the :py:obj:`spiders_reader`, :py:obj:`stairs_reader` instances of the :py:class:`.LinearReader` class, or an instance of a :py:class:`.TreeReader`, may convert the pregroup diagram into a monoidal form that is too complicated to be rendered properly in a text console. In these cases, diagrams cannot be displayed as text.

Rewrite rules and ansätze
-------------------------

.. note::
    Option only applicable to string diagrams.

The command-line interface supports all stages of the ``lambeq`` :ref:`pipeline <sec-pipeline>`, such as application of :term:`rewrite rules <rewrite rule>` and use of :term:`ansätze <ansatz (plural: ansätze)>` for converting the sentences into :term:`quantum circuits <quantum circuit>` or :term:`tensor networks <tensor network>`. For example, to read a file of sentences, parse them, apply the ``prepositional_phrase`` and ``determiner`` :term:`rewrite rules <rewrite rule>`, and use an :py:class:`.IQPAnsatz` with 1 :term:`qubit` assigned to sentence type, 1 :term:`qubit` to noun type, and 2 IQP layers, use the command:

.. code-block:: console

    $ lambeq -i sentences.txt -t -f image -g png
    >        -w prepositional_phrase determiner
    >        -a iqp -n dim_n=1 dim_s=1 n_layers=2
    >        -d image_folder

.. note::
    Since :term:`rewrite rules <rewrite rule>` and :term:`ansätze <ansatz (plural: ansätze)>` can produce output that is too complicated to be properly rendered in purely text form, text output in the console is not available for these cases.

For the classical case, applying a :py:class:`.SpiderAnsatz` with 2 dimensions assigned to sentence type and 4 dimensions to noun type, and the same rewrite rules as above, can be done with the following command:

.. code-block:: console

    $ lambeq -i sentences.txt -t -f image -g png
    >         -w prepositional_phrase determiner
    >         -a spider -n dim_n=4 dim_s=2
    >         -d image_folder

Other options
-------------

To store the :term:`DisCoPy` (for string diagrams) or the :py:class:`.CCGTree` objects (for the CCG trees) in ``json`` or ``pickle`` format, type:

.. code-block:: console

    $ lambeq -f pickle -i sentences.txt -o diagrams.pickle

or

.. code-block:: console

    $ lambeq -f json -i sentences.txt -o diagrams.json

Text output is also available with ascii-only characters:

.. code-block:: console

    $ lambeq -f text-ascii "John gave Mary a flower."

     John       gave      Mary    a    flower.
     ____  _____________  ____  _____  _______
      n    n.r s n.l n.l   n    n n.l     n   
      \_____/  |  |   \____/    |  \______/   
               |  \_____________/             

To avoid repeated long commands, arguments can be stored into a YAML file ``conf.yaml`` by adding an argument ``-y conf.yaml``.
To load the configuration from this file next time, ``-l conf.yaml`` can be added. Any arguments that were not provided in the command line will be taken from that file. If an argument is specified both in the command line and in the configuration file, the command-line argument takes priority.

.. _sec-detailed_options:

Detailed options
----------------

.. argparse::
   :filename: ../lambeq/cli.py
   :func: prepare_parser
   :prog: lambeq

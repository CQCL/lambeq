.. _sec-nlp-data:

Working with text data
======================

Datasets and corpora
--------------------

NLP work is heavily data-driven, and text data is organised into collections such as `datasets` and `corpora`. While sometimes these terms are (wrongly) used interchangeably, they differ in their purpose and structure.

A `dataset` is a structured collection of data that is designed for a specific task. In NLP, a dataset may consist of a set of labeled text documents that are used for training and evaluating a machine learning model. Each document in the dataset is labeled with a class or category that the model is trying to predict. For example, a dataset of movie reviews may be labeled with "positive" or "negative" sentiment, and a model can be trained to predict the sentiment of new, unlabeled reviews. Examples of datasets can be found in the folder `docs/examples/datasets <https://github.com/CQCL/lambeq/tree/main/docs/examples/datasets>`_ of the ``lambeq`` Github repository.

On the other hand, a `corpus` is an unstructured collection of text data that is designed for linguistic analysis. A corpus may consist of a large collection of text documents from a variety of sources, such as newspapers, books, and websites. The purpose of a corpus is to provide a representative sample of language use, which can be analysed to understand patterns in language structure and usage. An example of a corpus is the `British National Corpus (BNC) <http://www.natcorp.ox.ac.uk>`_, a 100-million word collection of samples of written and spoken language from a wide range of sources.

.. _sec-preprocessing:

Text pre-processing
-------------------

In order to prepare text data for analysis, NLP researchers use various pre-processing techniques. These are designed to convert raw text into a format that can be easily understood by machines. Some common pre-processing techniques include:

- **Tokenization**: This involves breaking down a text document into individual words or phrases, called tokens. This is typically the first step in text analysis. Tokenization is further discussed in :ref:`following section <sec-tokenization>`.
- **Stemming**: The process of reducing words to their root or stem form. This is done to reduce the number of unique words in a text document and to improve the efficiency of subsequent processing. For example, the words "programming", "programmer", and "programs" can all be reduced down to the common word stem "program".
- **Lemmatization**: Similar to stemming, but instead of reducing words to their root form, it reduces them to their base form or lemma (dictionary form). This can result in more accurate analysis, as it takes into account the context in which the word is used. For example, "run", "ran", and "runs" will be all mapped to the lemma "run", removing any inflections but respecting the part-of-speech of the word.
- **Stop-word removal**: Stop words are common words that are sometimes removed from text documents as they do not carry much meaning. Examples of stop words include determiners (e.g. "a", "the"), auxiliary verbs (e.g. "am", "was"), prepositions (e.g. "in", "at"), and conjunctions (e.g. "and", "but", "or").
- **Part-of-Speech (POS) tagging**: This involves labeling each word in a text document with its corresponding part of speech, such as noun, verb, or adjective. This can be useful for identifying the role of each word in a sentence and for extracting meaningful information from a text document. For example, the words in the sentence "John gave Mary a flower" would be labeled as "John_N gave_VB Mary_N a_DET flower_N".

It is important to note that with the advent of deep learning and the increase of computational power, some of these pre-processing steps have become less useful in practice. For example, deep learning models are capable of automatically learning and identifying the important features and patterns within the raw text data, making the need for certain pre-processing steps such as stemming and stop-word removal redundant. It is important, however, to note that these pre-processing steps may still be useful in certain specific scenarios, such as when dealing with limited training data or when working with domain-specific languages.

.. _sec-tokenization:

Tokenization
------------

Tokenization is the process of breaking down a text or sentence into smaller units called tokens. Tokens are the building blocks of natural language processing, and they are typically words, punctuation marks, or other meaningful elements of a sentence. The purpose of tokenization is to make it easier for computers to process human language, by providing a structured representation of text data that can be analysed, searched, and manipulated.

Tokenization comes in many different forms. Some examples are the following:

.. _wordtok:

- **Word tokenization:** In this very common form of tokenization, a sentence is split into individual words or tokens. For example, the sentence "The quick brown fox jumps over the lazy dog" would be tokenized into the following list of words:

  .. code-block:: console

    ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]

  In a more complete example, consider a sentence that includes various punctuation marks and contractions, such as "This sentence isn't worth £100 (or is it?).". The proper way to tokenize this sentence is the following, clearly separating every individual word and symbol:

  .. code-block:: console

    ["This", "sentence", "is", "n't", "worth", "£", "100",

     "(", "or", "is", "it", "?", ")", "."]

- **Sentence tokenization:** When working with paragraphs or documents, usually the first step is to split them into individual sentences. For example, the paragraph "I love pizza. It is my favorite food. I could eat it every day!" would be tokenized into the following list of sentences:

  .. code-block:: console

    ["I love pizza.", "It is my favorite food.", "I could eat it every day!"]

- **Phrase tokenization:** In this type of tokenization, a sentence is split into meaningful phrases or chunks. For example, the sentence "I want to book a flight to Paris" might be tokenized into the following phrases:

  .. code-block:: console

    ["I", "want to", "book", "a flight", "to", "Paris"]

.. _wordpiece:

- **Word-piece tokenization:**  A type of tokenization that breaks down words into their constituent morphemes, which are the smallest meaningful units of a word. Morphemes can be either words themselves or smaller units that carry meaning, such as prefixes, suffixes, and roots. Consider for example the sentence, "Unbelievable, I can't believe how amazing this is.". Word-piece tokenization would produce the following list of tokens:

  .. code-block:: console

     ["Un##believ##able", ",", "I", "can'", "t", "believe", "how", "amaz##ing", "this" "is."]

  In the example, the "##" symbols indicate that the subword is part of a larger word.

.. note::

  ``lambeq`` supports word and sentence tokenization through the :py:class:`.Tokeniser` class hierarchy and specifically the :py:class:`.SpacyTokeniser` class, based on the SpaCy package. For more information see :ref:`this detailed tutorial </tutorials/sentence-input.ipynb#Pre-processing-and-tokenisation>`.

Handling unknown words
----------------------

One of the most common challenges in NLP is the handling of unknown words, or `out-of-vocabulary` (OOV) words. The term refers to words that may appear during evaluation and testing, but they were not present in the training data of the model. One way to handle unknown words is to use :ref:`word-piece tokenization <wordpiece>`, which splits words into smaller subword units. This allows the model to learn representations for unseen words based on their subword units. For example, assume that word "unbelievable" does not appear in the training data, but the words "un##settl##ing", "believ##er", and "do##able" are present; the unknown word would still be able to be represented as a combination of individual word pieces, i.e. "un##believ##able".

When using :ref:`word tokenisation <wordtok>` (like in ``lambeq``), a common technique to handle unknown word is to introduce a special token ``UNK``. The method is based on the following steps:

1. Replace every rare word in the training data (e.g. every word that occurs less than a specified threshold, for example 3 times) with a special token ``UNK``.
2. During training, learn a representation for ``UNK`` as if there was any other token.
3. During evaluation, when you meet an unknown word, use the representation of ``UNK`` instead.

.. note::

  Note that in syntax-based models, such as :term:`DisCoCat`, handling unknown words with the above method becomes more complicated, since the type of each word needs to also be taken into account. In other words, you need to have a different ``UNK`` token for each grammatical type.
  ``lambeq`` simplifies this process by providing the :py:class:`~.UnknownWordsRewriteRule` which can be used to replace unknown words, and create a vocabulary from a set of diagrams.

.. rubric:: See also:

- :ref:`Pre-processing and tokenisation tutorial </tutorials/sentence-input.ipynb#Pre-processing-and-tokenisation>`
- :ref:`Tokenisation example notebook </examples/tokenisation.ipynb>`

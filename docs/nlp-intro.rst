.. _sec-nlp-intro:

Introduction
============

In this section we will briefly explore the field of Natural Language Processing and see how it is used for solving real-world problems related to human language. [#f1]_ Let's get started with some definitions.

NLP, QNLP and Computational Lingustics
--------------------------------------

:term:`Natural Language Processing <natural language processing (NLP)>` (NLP) is a field of AI that focuses on the interaction between computers and human language. NLP deals with the problems of generating and understanding natural language text and speech, in a way that is both effective and accurate. It involves a range of methods and techniques, including statistical and rule-based approaches, machine learning, and deep learning. These methods are used to develop NLP models and systems that can perform various language-related tasks, such as text classification, sentiment analysis, speech recognition, machine translation, and many others.

`Computational linguistics`, on the other hand, is a broader field that encompasses NLP, and in principle deals with the study of human language and its structure from a computational perspective. It uses algorithms, models, and mathematical theories to analyze and understand human language, and to develop natural language processing systems. While NLP focuses more on the practical application of computational linguistics techniques to solve real-world problems related to human language, computational linguistics is more concerned with the theoretical study of language and its formal properties.

.. note::

    ``lambeq`` is a language modelling tool capable of representing language in many different levels of abstraction, such as syntax trees, pregroup diagrams, string/monoidal diagrams, tensor networks and quantum circuits, and for this reason, it is conceptually closer to (quantum) computational linguistics than merely to practical NLP.

NLP is one of the most important fields of AI, since the ability to understand and use language is one of the defining features of human intelligence; thus, being able to teach machines this skill is a significant step towards developing AI. In fact, the famous `Turing test <https://en.wikipedia.org/wiki/Turing_test>`_ for AI involves determining whether a machine can exhibit intelligent behaviour that is indistinguishable from that of a human, solely based on the effective use of language.

Having defined the purpose and scope of NLP as above, `Quantum NLP` (QNLP) is simply NLP on quantum computers. More specifically, QNLP is aimed at the design and implementation of NLP models that exploit certain quantum phenomena such as superposition, entanglement, and interference to perform language-related tasks on quantum hardware. By applying quantum principles to language processing, QNLP seeks to provide a more holistic and accurate model of communication that can capture the nuances and complexities of human language better than traditional "classical" models.

.. _sec-nlp-tasks:

Tasks and applications
----------------------

There are numerous important applications of NLP across various industries and domains. Some of the most prominent ones include:

- **Chatbots and virtual assistants**: NLP is widely used in chatbots and virtual assistants, enabling them to understand natural language queries and respond accordingly.
- **Sentiment analysis**: The task of analyzing social media data, customer feedback, and product reviews to determine sentiment and gain insights into customer preferences.
- **Machine translation**: NLP can be used to enable accurate and efficient translation of text between different languages.
- **Speech recognition**: Speech recognition systems can transcribe spoken language into text, enabling voice-controlled applications.
- **Named Entity Recognition**: Techniques used to identify and extract entities such as people, organizations, and locations from text.
- **Text summarization**: The task of summarizing large volumes of text, making it easier to process and understand.
- **Information retrieval**: NLP is used in search engines and recommendation systems to enable relevant and accurate results based on natural language queries.

Typical NLP workflow
--------------------

In this section, we examine the sequence of steps involved in processing and analyzing natural language data. While the exact workflow may vary depending on the specific task and dataset, there are several common steps that are typically involved.

#. **Data collection:** The first step in any NLP project is to collect the relevant data. This may involve web scraping, accessing APIs, or using pre-existing datasets. It is important to ensure that the data is of high quality and properly formatted.
#. **Text preprocessing:** Once the data has been collected, the next step is to preprocess the text. This involves several steps such as tokenization, stopword removal, stemming or lemmatization, and part-of-speech tagging. The goal of preprocessing is to convert raw text into a structured format that can be used for analysis. More information can be found :ref:`here <sec-preprocessing>`.
#. **Text representation:** After preprocessing, the text data needs to be represented in a format that can be used for analysis. This typically involves using word embeddings, pre-trained language models such as BERT or GPT-3, or :term:`bag-of-words` models.
#. **Model training:** With the text data represented in a suitable format, the next step is to train a model. Depending on the task, this may involve using machine learning algorithms such as logistic regression or neural networks. The model is trained on a labeled dataset and validated on a held-out dataset to ensure that it generalizes well (see :ref:`sec-ml`).
#. **Model evaluation:** Once the model is trained, it needs to be evaluated to determine how well it performs on unseen data. This involves using :ref:`evaluation metrics <sec-evaluation>` such as accuracy, precision, recall, and F1 score. It is important to ensure that the model performs well on both the training and validation data, as well as on a test dataset (see :ref:`sec-ml`).

In the following sections, we will focus on some important text pre-processing concepts and techniques.

.. [#f1] This tutorial has been created with the help of `ChatGPT <https://openai.com/blog/chatgpt>`_.
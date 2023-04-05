.. _sec-ml:

Machine learning best practices
===============================

In NLP and machine learning, following careful evaluation methods is crucial to ensure that the model is performing well on unseen data and to avoid overfitting. One important step in evaluation is to split the dataset into training, validation, and test sets. The training set is used to train the model, the validation set is used to tune the model's hyperparameters, and the test set is used to evaluate the final performance of the model. More specifically:

- **Training set:** This is the largest portion of your data, and it is used to train your model. Your model will learn from the patterns in this data.
- **Development set:** This set is also known as a `validation set`. It is used to tune your model's hyperparameters and ensure that your model is not overfitting to the training data. You will use this set to evaluate your model's performance during training and make adjustments as needed.
- **Test set:** This set is used to evaluate your model's performance after training is complete. You should never use this data during training or hyperparameter tuning, as doing so could cause overfitting.

The typical split for these sets is 60% training, 20% development, and 20% testing, but the exact split can depend on the size of your dataset and the complexity of your model.

Additionally, it is important to choose the right :ref:`evaluation metric <sec-evaluation>` for your model. Different models may require different metrics to evaluate their performance, so it is important to understand the strengths and weaknesses of each metric and choose the one that is most appropriate for your use case.

Cross-validation
----------------
`Cross-validation` is a technique used to evaluate the performance of a machine learning model by partitioning the data into multiple subsets, or `folds`, and training the model on different subsets while using the remaining fold(s) for validation.

The basic idea behind cross-validation is to use multiple samples of the data for training and validation to get a more accurate estimate of the model's performance on new, unseen data. By using multiple folds, the model can be evaluated on a variety of data samples, which can help to identify any potential issues with overfitting or bias.

There are several different types of cross-validation techniques, including:

- **k-fold cross-validation:** In this approach, the data is partitioned into `k` equally-sized folds. The model is trained on :math:`k-1` folds and the remaining fold is used for validation. This process is repeated `k` times, with each fold used for validation exactly once.
- **Stratified k-fold cross-validation:** This technique is similar to `k`-fold cross-validation, but it ensures that the distribution of classes in each fold is similar to the overall distribution in the full dataset. This can be useful for datasets with imbalanced classes.
- **Leave-one-out cross-validation:** In this technique, the data is partitioned into `n` folds, where `n` is the number of samples in the dataset. The model is trained on all but one sample, which is used for validation. This process is repeated `n` times, with each sample used for validation exactly once.

Cross-validation can help to ensure that a model is not overfitting to the training data and can provide a more accurate estimate of the model's performance on new, unseen data. It can also be used to compare the performance of different models or hyperparameters. However, it can be computationally expensive and may not be necessary for smaller datasets or less complex models.

.. note::

   Training in ``lambeq`` is handled by the :py:mod:`~.training` package, which provides a detailed hierarchy of classes aimed at supervised learning, as well as the means for collaboration with popular ML and QML libraries such as PyTorch and PennyLane.

.. rubric:: See also:

- :ref:`Training tutorial <sec-training>`

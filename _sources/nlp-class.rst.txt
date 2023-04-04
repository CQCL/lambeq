Text classification
===================

One of the most fundamental tasks in NLP is text classification, which involves categorising textual data into predefined categories. It plays a vital role in a variety of NLP applications, including sentiment analysis, spam detection, topic modeling, and language identification, among others. By categorising texts into relevant categories, machines can analyse and derive insights from large volumes of textual data, making it possible to automate decision-making processes and perform tasks that would otherwise be time-consuming or impossible for humans to do.

Binary vs multi-class classification
------------------------------------

Binary classification and multi-class classification involve assigning a label or category to an input data point. In `binary classification`, there are only two possible output categories, and the goal is to classify input data points into one of these two categories. For example, classifying emails as spam or not spam.

On the other hand, `multi-class classification` involves assigning a data point to one of more than two possible output categories. For example, classifying images of animals into categories such as cats, dogs, and birds.

Multi-class classification problems can be further divided into two subcategories: multi-class `single-label` classification and multi-class `multi-label` classification. In multi-class single-label classification, each input data point is assigned to one and only one output category. In contrast, in multi-class multi-label classification, each input data point can be assigned to one or more output categories simultaneously.

In general, binary classification is a simpler and more straightforward problem to solve than multi-class classification, but multi-class classification problems are more representative of real-world scenarios where there are multiple possible categories to that a data point could belong.

Loss functions
--------------

For binary classification tasks, the loss function of choice is binary cross-entropy. Below, :math:`y_i` is the true label for the :math:`i` th data point, :math:`p(y_i)` represents the probability that the model assigns to the specific label, and :math:`N` is the number of data points.

.. math::

   H(p, q) = -\frac{1}{N}\sum_{i=1}^N [y_i \log(p(y_i)) + (1-y_i) \log(1-p(y_i))]

For multi-class classification, the loss function is usually the categorical version of cross-entropy. Here, :math:`M` is the number of classes, :math:`p(x_i)` is the true probability for the :math:`i` th class, and :math:`q(x_i)` the probability predicted by the model.

.. math::

   H(p, q) = -\sum_{i=1}^M p(x_i) \log(q(x_i))

.. note::

   ``lambeq`` provides a number of loss functions that can be used out-of-the-box during training, such as :py:class:`~.BinaryCrossEntropyLoss`, :py:class:`~.CrossEntropyLoss`, and :py:class:`~.MSELoss`.

.. _sec-evaluation:

Evaluation metrics
------------------

The most common metrics to evaluate the performance of classification models is accuracy, precision, recall, and F-score. Each metric has its own strengths and weaknesses, and can be useful in different contexts.

- `Accuracy` is usually the standard way to evaluate classification, and it measures how often the model correctly predicts the class of an instance. It is calculated as the ratio of correct predictions to the total number of predictions. This metric can be useful when the classes in the dataset are balanced, meaning that there are roughly equal numbers of instances in each class. In this case, accuracy can provide a good overall measure of how well the model is performing.

.. math::
   \text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{True Positives} + \text{True Negatives} + \text{False Positives} + \text{False Negatives}}

- `Precision` is the proportion of true positive predictions among all positive predictions. It is expressed as the ratio of true positives to the total number of instances that the model predicts as positive. Precision is useful when the cost of false positives is high, such as in spam filtering or legal decision making.

.. math::

   \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}

- `Recall`, also known as `sensitivity`, is the proportion of true positive predictions among all actual positive instances in the dataset. Recall is calculated as the ratio of true positives to the total number of instances of that class. It can be helpful when the goal of the model is to identify all instances of a particular class, such as in medical diagnosis or fraud detection.

.. math::

   \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}

These two measures can be competing in the sense that increasing precision can decrease recall and vice versa. This trade-off occurs because precision and recall measure different aspects of the model's performance. High precision means that the model is accurate in its positive predictions, but it may miss some true positive instances, leading to lower recall. On the other hand, high recall means that the model identifies most of the positive instances, but it may have more false positives, leading to lower precision.

To address this, researchers use `F-score`, also known as the `F1` score, which is a combined measure of precision and recall. It is calculated as the harmonic mean of precision and recall and provides a way to balance these two metrics. F-score is useful when both precision and recall are important and can be used to compare models that have different tradeoffs between these two metrics.

.. math::

   \text{F-score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}

.. note::

   For examples of text classification with ``lambeq``, see the :ref:`Training tutorial <sec-training>`.

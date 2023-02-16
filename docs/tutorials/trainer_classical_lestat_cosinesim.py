#!/usr/bin/env python
# coding: utf-8

# # Training: Classical case

# ## Preparation
# 
# We start with importing PyTorch and specifying some training hyperparameters.

# In[1]:
from pytorch_trainer_cosinesim import PytorchTrainerCosineSim
from lambeq import PytorchModel
import torch
import logging
logging.basicConfig(level=logging.INFO)

BATCH_SIZE = 30
EPOCHS = 30
LEARNING_RATE = 3e-2
SEED = 0


# ### Input data
# 
# Let's read the data and print some example sentences.

# In[2]:


class MyCustomModel(PytorchModel):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(2, 100)

    def forward(self, input):
        """define a custom forward pass here"""
        preds = self.get_diagram_output(input)
        preds = self.net(preds)
        return preds


def read_data(filename):
    labels, sentences = [], []
    with open(filename) as f:
        for line in f:
            line_split=line.split("\t")
            t = float(line_split[0])
            labels.append([t])
            sentences.append(line_split[1].strip())
    return labels, sentences

logging.info("before reading data")
train_labels, train_data_claim = read_data('../examples/datasets/lestat_small_train_data_claim.txt')
train_labels, train_data_evidence = read_data('../examples/datasets/lestat_small_train_data_evidence.txt')
val_labels, val_data = read_data('../examples/datasets/lestat_small_dev_data.txt')
test_labels, test_data = read_data('../examples/datasets/lestat_small_test_data.txt')
logging.info("after reading data")

# In[3]:


#train_data[:5]


# Targets are represented as 2-dimensional arrays:

# In[4]:


train_labels[:5]


# ### Creating and parameterising diagrams

# In[5]:


# from lambeq import BobcatParser
# parser = BobcatParser(verbose='text')
# train_diagrams_claim = parser.sentences2diagrams(train_data_claim)
# train_diagrams_evidence = parser.sentences2diagrams(train_data_claim_evidence)
# val_diagrams = parser.sentences2diagrams(val_data)
# test_diagrams = parser.sentences2diagrams(test_data)

#
logging.info("before converting sentence to diagrams")
from lambeq import spiders_reader
train_diagrams_claim = [spiders_reader.sentence2diagram(sent) for sent in train_data_claim]
train_diagrams_evidence = [spiders_reader.sentence2diagram(sent) for sent in train_data_evidence]
val_diagrams = [spiders_reader.sentence2diagram(sent) for sent in val_data]
test_diagrams = [spiders_reader.sentence2diagram(sent) for sent in test_data]
train_diagrams_evidence[0].draw(figsize=(13,6), fontsize=12)

logging.info("after converting sentence to diagrams")



# In[6]:


from discopy import Dim

from lambeq import AtomicType, SpiderAnsatz
logging.info("before ansatz")
ansatz = SpiderAnsatz({AtomicType.NOUN: Dim(2),
                       AtomicType.SENTENCE: Dim(2)})

train_circuits_claim = [ansatz(diagram) for diagram in train_diagrams_claim]
train_circuits_evidence = [ansatz(diagram) for diagram in train_diagrams_evidence]

val_circuits =  [ansatz(diagram) for diagram in val_diagrams]
test_circuits = [ansatz(diagram) for diagram in test_diagrams]
logging.info("after ansatz")
#train_circuits[0].draw()

# ## Training
# 
# ### Instantiate model

# In[7]:

logging.info("going to load model")


all_circuits = train_circuits_claim +val_circuits + test_circuits
all_labels=train_labels+val_labels+test_labels
#model = PytorchModel.from_diagrams(all_circuits)


custom_model = MyCustomModel.from_diagrams(all_circuits)

# ### Define evaluation metric
# 
# Optionally, we can provide a dictionary of callable evaluation metrics with the signature ``metric(y_hat, y)``.

# In[8]:

logging.info("after loading model")
sig = torch.sigmoid

def accuracy(y_hat, y):
    print(f"y_hat={y_hat}")
    return torch.sum(torch.eq(torch.round(sig(y_hat)), y))/len(y)/2  # half due to double-counting

eval_metrics = {"acc": accuracy}


# ### Initialise trainer

# In[9]:
logging.info("before calling trainer")




trainer = PytorchTrainerCosineSim(
        model=custom_model,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.AdamW,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        evaluate_functions=eval_metrics,
        evaluate_on_train=True,
        verbose='text',
        seed=SEED)


# ### Create datasets

# In[10]:

logging.info("before calling dataset")
from dataset_mithun import Dataset
input_data=(train_circuits_claim, train_circuits_evidence)
logging.info("length of data[0]=%s",len(input_data[0]))
logging.info("length of targets=%s",len(train_labels))
train_dataset = Dataset(
            input_data,
            train_labels,
            batch_size=BATCH_SIZE)



#logging.info("after calling trainer before dataset")
#val_dataset = Dataset(val_circuits, val_labels, shuffle=False)


# ### Train

# In[11]:
logging.info("after loading datasets . before trainer.fit")

trainer.fit(train_dataset, train_dataset, evaluation_step=1, logging_step=5)
logging.info("after e trainer.fit")

# ## Results
# 
# Finally, we visualise the results and evaluate the model on the test data.

# In[12]:


import matplotlib.pyplot as plt

fig1, ((ax_tl, ax_tr), (ax_bl, ax_br)) = plt.subplots(2, 2, sharey='row', figsize=(10, 6))

ax_tl.set_title('Training set')
ax_tr.set_title('Development set')
ax_bl.set_xlabel('Epochs')
ax_br.set_xlabel('Epochs')
ax_bl.set_ylabel('Accuracy')
ax_tl.set_ylabel('Loss')

colours = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
ax_tl.plot(trainer.train_epoch_costs, color=next(colours))
ax_bl.plot(trainer.train_results['acc'], color=next(colours))
ax_tr.plot(trainer.val_costs, color=next(colours))
ax_br.plot(trainer.val_results['acc'], color=next(colours))

# print test accuracy
test_acc = accuracy(model(test_circuits), torch.tensor(test_labels))
print('Test accuracy:', test_acc.item())


# ## Adding custom layers to the model

# In[13]:


# The rest follows the same procedure as explained above, i.e. initialise a trainer, fit the model and visualise the results.

# In[14]:


custom_model = MyCustomModel.from_diagrams(all_circuits)
custom_model_trainer = PytorchTrainer(
        model=custom_model,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.AdamW,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        evaluate_functions=eval_metrics,
        evaluate_on_train=True,
        verbose='text',
        seed=SEED)
custom_model_trainer.fit(train_dataset, val_dataset, logging_step=5)


# In[15]:


import matplotlib.pyplot as plt

fig1, ((ax_tl, ax_tr), (ax_bl, ax_br)) = plt.subplots(2, 2, sharey='row', figsize=(10, 6))

ax_tl.set_title('Training set')
ax_tr.set_title('Development set')
ax_bl.set_xlabel('Epochs')
ax_br.set_xlabel('Epochs')
ax_bl.set_ylabel('Accuracy')
ax_tl.set_ylabel('Loss')

colours = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
ax_tl.plot(custom_model_trainer.train_epoch_costs, color=next(colours))
ax_bl.plot(custom_model_trainer.train_results['acc'], color=next(colours))
ax_tr.plot(custom_model_trainer.val_costs, color=next(colours))
ax_br.plot(custom_model_trainer.val_results['acc'], color=next(colours))

# print test accuracy
test_acc = accuracy(model(test_circuits), torch.tensor(test_labels))
print('Test accuracy:', test_acc.item())


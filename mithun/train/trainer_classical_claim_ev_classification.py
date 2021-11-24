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
from discopy import grammar
logging.basicConfig(level=logging.INFO)
from mithun.utils.utils import *
from mithun.utils.dataset_mithun import Dataset
from lambeq import BobcatParser, SpacyTokeniser



config = read_config()



# ### Input data
# 
# Let's read the data and print some example sentences.

# In[2]:


class MyCustomModel(PytorchModel):
    def __init__(self):
        super().__init__()

        self.net = torch.nn.Linear(2, config['LAST_LAYER_NEURON_COUNT'])

    def forward(self, input):
        """define a custom forward pass here"""
        preds = self.get_diagram_output(input)
        preds = self.net(preds)
        return preds


train_labels, train_data_claim = read_data_float_label(get_full_path(config['BASE_PATH_DATA'],config['MNLI_TRAIN_LAMBEQ_FORMAT_CLAIM_SMALL']))
train_labels, train_data_evidence = read_data_float_label(get_full_path(config['BASE_PATH_DATA'],config['MNLI_TRAIN_LAMBEQ_FORMAT_CLAIM_SMALL']))


val_labels, val_data_claim = read_data_float_label(get_full_path(config['BASE_PATH_DATA'],config['SNLI_DEV_LAMBEQ_FORMAT_CLAIM_SMALL']))
val_labels, val_data_evidence = read_data_float_label(get_full_path(config['BASE_PATH_DATA'],config['SNLI_DEV_LAMBEQ_FORMAT_EVIDENCE_SMALL']))


#test_labels, test_data = read_data_float_label(get_full_path(config['BASE_PATH_DATA'],config['SNLI_TRAIN_LAMBEQ_FORMAT_CLAIM_SMALL']))
logging.debug("after reading data")

# In[3]:


#train_data[:5]


# Targets are represented as 2-dimensional arrays:

# In[4]:


train_labels[:5]


# ### Creating and parameterising diagrams

# In[5]:

if(config['TYPE_OF_MODEL']=='discocat'):

    parser = BobcatParser(verbose='text')
    tokeniser = SpacyTokeniser()
    train_diagrams_claim = parser.sentences2diagrams(tokeniser.tokenise_sentences(train_data_claim),verbose='text',tokenised=True)

    train_diagrams_evidence = parser.sentences2diagrams(tokeniser.tokenise_sentences(train_data_evidence), verbose='text',
                                                     tokenised=True)
    if config['DRAW']:
        grammar.draw(train_diagrams_claim[0], figsize=(14, 3), fontsize=12)
    val_diagrams_claim = parser.sentences2diagrams(tokeniser.tokenise_sentences(val_data_claim), verbose='text',
                                                     tokenised=True)

    val_diagrams_evidence = parser.sentences2diagrams(tokeniser.tokenise_sentences(val_data_evidence),
                                                        verbose='text',
                                                        tokenised=True)

if(config['TYPE_OF_MODEL']=='spider'):
    logging.debug("before converting sentence to diagrams")
    from lambeq import spiders_reader
    train_diagrams_claim = [spiders_reader.sentence2diagram(sent) for sent in train_data_claim]
    train_diagrams_evidence = [spiders_reader.sentence2diagram(sent) for sent in train_data_evidence]
    val_diagrams_claim = [spiders_reader.sentence2diagram(sent) for sent in val_data_claim]
    val_diagrams_evidence = [spiders_reader.sentence2diagram(sent) for sent in val_data_evidence]
    if config['DRAW']:
        train_diagrams_evidence[0].draw(figsize=(13,6), fontsize=12)

logging.debug("after converting sentence to diagrams")



# In[6]:


from discopy import Dim

from lambeq import AtomicType, SpiderAnsatz
logging.debug("before ansatz")
ansatz = SpiderAnsatz({AtomicType.NOUN: Dim(2),
                       AtomicType.SENTENCE: Dim(2),
                       AtomicType.PREPOSITIONAL_PHRASE: Dim(2)
                       })

train_circuits_claim = [ansatz(diagram) for diagram in train_diagrams_claim]
train_circuits_evidence = [ansatz(diagram) for diagram in train_diagrams_evidence]

val_circuits_claim = [ansatz(diagram) for diagram in val_diagrams_claim]
val_circuits_evidence = [ansatz(diagram) for diagram in val_diagrams_evidence]

logging.debug("after ansatz")
if config['DRAW']:
    train_circuits_claim[0].draw()

# ## Training
# 
# ### Instantiate model

# In[7]:

logging.debug("going to load model")


all_circuits = train_circuits_claim +train_circuits_evidence+val_circuits_claim+val_circuits_evidence
all_labels=train_labels+val_labels
#model = PytorchModel.from_diagrams(all_circuits)


custom_model = MyCustomModel.from_diagrams(all_circuits)

# ### Define evaluation metric
# 
# Optionally, we can provide a dictionary of callable evaluation metrics with the signature ``metric(y_hat, y)``.

# In[8]:

logging.debug("after loading model")
sig = torch.sigmoid

def accuracy(y_hat, y):
    return torch.sum(torch.eq(torch.round(sig(y_hat)), y))/len(y)/2  # half due to double-counting

eval_metrics = {"acc": accuracy}


# ### Initialise trainer

# In[9]:



logging.debug(type(config['LEARNING_RATE']))
logging.debug("before calling trainer")
trainer = PytorchTrainerCosineSim(
        model=custom_model,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.AdamW,
        learning_rate=config['LEARNING_RATE'],
        epochs=config['EPOCHS'],
        evaluate_functions=eval_metrics,
        evaluate_on_train=True,
        verbose='text',
        seed=config['SEED'])


# ### Create datasets

# In[10]:

logging.debug("before calling dataset")

input_data=(train_circuits_claim, train_circuits_evidence)
logging.debug("length of data[0]=%s",len(input_data[0]))
logging.debug("length of targets=%s",len(train_labels))
train_dataset = Dataset(
            input_data,
            train_labels,
            batch_size=config['BATCH_SIZE'])



logging.debug("after calling trainer before dataset")
#val_dataset = Dataset(val_circuits, val_labels, shuffle=False)


# ### Train

# In[11]:
logging.debug("after loading datasets . before trainer.fit")

trainer.fit(train_dataset, train_dataset, evaluation_step=1, logging_step=5)
logging.debug("after e trainer.fit")

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

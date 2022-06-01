### module contains methods to prepare data for a transfer learning model
### adopt hugging face text classifcation model

import datasets
from datasets import Dataset
from sklearn.model_selection import train_test_split

from transformers import AutoTokeniner
from transformers import create_optimizer
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, TextClassificationPipeline



def df_to_ds(df, train_size= 0.7):
    '''
    Convert pandas dataframe to huggingface dataset.
    Dataframe should have 2 columns - job description (raw text) and category (strings).
    Output dataset includes "train" and "test" subsets, with "job_description" and
    "label" in each subset.
    '''

    category_dict = {'analyst junior': 0,
                        'analyst mid-level': 2,
                        'analyst senior': 1,
                        'engineer junior': 7,
                        'engineer mid-level': 8,
                        'engineer senior': 5,
                        'scien junior': 6,
                        'scien mid-level': 4,
                        'scien senior': 3}

    temp = df[['job_description','Category']]

    temp['label']= temp["Category"].map(category_dict)

    temp.drop(columns= 'Category', inplace= True)

    train, test = train_test_split(temp, train_size= train_size)

    train_dataset = Dataset.from_dict(train)

    test_dataset = Dataset.from_dict(test)

    return datasets.DatasetDict({"train":train_dataset,
                                "test":test_dataset})


class transfer_learning_trainer():
    '''Class to finetune DistilBERT model on job description data to determine
    what category a job description belongs to'''

    def __init__(self, dataset, epochs = 3):
        self.model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                                          num_labels=9)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.ds = dataset
        self.epochs = epochs

    def preprocess_function(self, examples):
        '''
        Return tokenized values of a job description.
        Truncate sequences to be no longer than DistilBERT's max input length.
        '''
        return self.tokenizer(examples["job_description"], truncation=True)

    def tl_train_test_sets(self):
        '''
        Tokenize, pad and divide datasets into train and test sets for model input.
        Tokenizer from pretrained "distilbert-base-uncased"
        '''

        # Load the DistilBERT tokenizer to process the text field:
        self.tokenized_jobs = self.ds.map(self.preprocess_function, batched = True)

        # Use DataCollatorWithPadding to create a batch of examples
        data_collator = DataCollatorWithPadding(tokenizer= self.tokenizer, return_tensors="tf")

        # create padded & tokenized train and test sets
        self.tf_train_set = self.tokenized_jobs['train'].to_tf_dataset(
            columns=["attention_mask", "input_ids", "label"],
            shuffle=True,
            batch_size=16,
            collate_fn=data_collator,
        )

        self.tf_validation_set = self.tokenized_jobs["test"].to_tf_dataset(
            columns=["attention_mask", "input_ids", "label"],
            shuffle=False,
            batch_size=16,
            collate_fn=data_collator,
        )

        return self

    def run(self):
        '''
        Compile model and fit on train dataset
        '''
        batch_size = 16
        num_epochs = 5
        batches_per_epoch = len(self.tonkenized_jobs['train']) // batch_size
        total_train_steps = int(batches_per_epoch * num_epochs)
        optimizer, schedule = create_optimizer(init_lr=2e-5,
                                            num_warmup_steps=0,
                                            num_train_steps=total_train_steps)

        self.model.compile(optimizer=optimizer)
        self.model.fit(x= self.tf_train_set,
                       validation_data= self.tf_validation_set,
                       epochs=self.epochs)
        return self.model

    def tokenizer_with_truncation(self, text, **kwargs):
        return self.tokenizer(text, truncation= True, **kwargs)

    def predict(self,text):
        pipe = TextClassificationPipeline(model= self.model,
                                            tokenizer= self.tokenizer_with_truncation)

        return pipe(text)

    def compute_accuracy(self,X_test, y_test):

        # list of labels in integer format
        # predict method returns label (string) and corresponding score
        y_pred = [ int(_['label'][6]) for _ in self.predict(X_test)]

        # instantiate tensorflow accuracy metrics
        m = tf.keras.metrics.Accuracy()

        # reset if used previously
        m.reset_state()

        m.update_state(y_test, y_pred)
        return m.result().numpy()

<p align="center">
  <a href="#"><img src="./misc/quora.png" width="350"></a></br>
  by</br>
  <a href="#"><img src="./misc/Kaggle_logo.png" width="60"></a>
</p>



We use the `BiLSTM attention Kfold add features` kernel to reach [0.703 score](https://www.kaggle.com/c/quora-insincere-questions-classification/leaderboard) at the Kaggle Quora competition.  
This kernel stands on : 
- [gru-capsule](https://www.kaggle.com/gmhost/gru-capsule)
- How to: [Preprocessing when using embedding](https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings)
- Improve your Score with some [Text Preprocessing](https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing)
- Simple [attention layer](https://github.com/mttk/rnn-classifier/blob/master/model.py)
- [baseline-pytorch-bilstm](https://www.kaggle.com/ziliwang/baseline-pytorch-bilstm)
- [pytorch-starter](https://www.kaggle.com/hengzheng/pytorch-starter)

## Basic Parameters

| name         | value  | 
|--------------|--------|
| embed_size   | 300    |
| max_features | 120000 |
| maxlen       | 70     |
| batch_size   | 512    |
| n_epochs     | 5      |
| n_splits     | 5      |

## Ensure determinism in the results 
`seed_everything` : A common headache in this competition is the lack of determinism in the results due to cudnn. This [Kerne](https://www.kaggle.com/hengzheng/pytorch-starter) has a solution in Pytorch.  



## Code for Loading Embeddings
Function from [here](https://www.kaggle.com/gmhost/gru-capsule).
- `load_glove`
- `load_fasttext`
- `load_para`

## Load processed training data from disk
- `build_vocab`

## Normalization

Borrowed from:
- How to: [Preprocessing when using embeddings](https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings)
- Improve your Score with some [Text Preprocessing](https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing)

- `build_vocab`
- `known_contractions`
- `clean_contractions`
- `correct_spelling`
- `unknown_punct`
- `clean_numbers`
- `clean_special_chars`
- `add_lower`
- `clean_text`
- `clean_numbers`
- `_get_mispell`
- `replace_typical_misspell`

Extra feature part taken [here](https://github.com/wongchunghang/toxic-comment-challenge-lstm/blob/master/toxic_comment_9872_model.ipynb)

- `add_features_before_cleaning`
    - `count_contains_a_punct`
    - `count_contains_a_string`
    - `count_words_more_frequent_in_insc`
    - `count_words_more_frequent_in_sc`

- `add_features_custom`
    - `count_contains_a_string`
    - `count_words_more_frequent_in_insc`
    - `count_words_more_frequent_in_sc`

`add_features`

`load_and_prec`
    - `lower`
    - `Clean the text`
    - `Clean numbers`
    - `Clean speelings`
    - `fill up the missing values`

Add Features
- https://github.com/wongchunghang/toxic-comment-challenge-lstm/blob/master/toxic_comment_9872_model.ipynb
- Tokenize the sentences
- Pad the sentences 
- Get the target values
- Splitting to training and a final test set
- shuffling the data
- fill up the missing values


## Save data on disk

## Load dataset from disk


## Load Embeddings

Two embedding matrices have been used. Glove, and paragram. The mean of the two is used as the final embedding matrix.
Missing entries in the embedding are set using np.random.normal so we have to seed here too


## Use Stratified K Fold to improve results


## Cyclic CLR

Code taken [here](https://www.kaggle.com/dannykliu/lstm-with-attention-clr-in-pytorch)
code inspired from: https://github.com/anandsaha/pytorch.cyclic.learning.rate/blob/master/cls.py
- `CyclicLR`
    - `batch_step`
    - `_triangular_scale_fn`
    - `_triangular2_scale_fn`
    - `_exp_range_scale_fn`
    - `get_lr`


## Model Architecture

Binary LSTM with an attention layer and an additional fully connected layer. Also added extra features taken from a winning kernel of the toxic comments competition. Also using CLR and a capsule Layer. Blended together in concatentation.

Initial idea borrowed from: https://www.kaggle.com/ziliwang/baseline-pytorch-bilstm
- `Embed_Layer`
    - `forward`

- `GRU_Layer`
    - `init_weights`
    - `forward`

- `Caps_Layer`
    - `forward`
    - `squash`

- `Capsule_Main`
    - `forward`

- `Attention`
    - `forward`

- `NeuralNet`
    - `forward`

## Training

The method for training is borrowed from https://www.kaggle.com/hengzheng/pytorch-starter
- `MyDataset`
    -`__getitem__`
    - `__len__`

- `sigmoid`


## Find final Thresshold

Borrowed from: https://www.kaggle.com/ziliwang/baseline-pytorch-bilstm
- `bestThresshold`

## submission 

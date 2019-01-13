# Twitter-Sentiment-Analysis
An example of sentiment analysis on Twitter using Emotions to label the sentiment of the tweet.

To complete the analysis exploits two python libraries:

- [scikit-learn](https://scikit-learn.org)
- [Pandas](https://pandas.pydata.org/)

# Pre Processing

The preprocessing scripts modifies the tweets content in order to make possible the further analysis. 

- Any url is removed.
- Any @Username is removed.
- Any additional white spaces is removed
- Ant not alphanumeric symbol is removed 
- Hashtags are substituted with the corresponding word

# Classifier

The actual analysis happens by the means of the TrainingModel.py script.

It requires as arguments (the order is relevant):

- the file of train.csv.

The analysis follows the following steps:

1) After reading the train.csv it will seperate the sentiments and sentiment_text. 
2) The tweets dataset which are now separated will be merged and divides into a train dataset, which will be used to train the classifier, and a test dataset, used to test it. Respectively there will be the 85% of the tweets for training and the 15% for testing.
3) From each tweet will be extracted a feature-set
4) We use NLTK to describe each tweets in terms of the features it contains. Indeed, we create a list of words ordered by frequency.
5) We trains a RandomForestClassifier with such a dataset
6) We test the classifier using the remaining 15% of the tweets and we process in the same way of the training dataset.


# Results

1) Run the Flask file index.py
2) goto browser and write URL http://localhost:5000 

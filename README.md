# CourseProject

**Quick Note: If you want to run this source code yourself, you will need to gain Elevated Access to the Tweepy API to make API calls in quick succession. The provided Colab Notebook and Demo will contain all source code already ran to show results of each respective code block (this is also to ease grading workflow). Also: Please make sure all files are kept in their respective folders and downloaded to drive as we use drive.mount for both our Logisitical Regression models(keep/run everything from "CS410Project", as well as our Data Visualizations (database folder)**

This project repo covers the CS410 Text Information Systems Final Project conducted by our 4 man team. The main idea of this repo is to provide code, documentation, and a demo for a Twitter Sentiment Analysis Tool based on Crypto market data using Python, Tweepy, TextBlob for initial sentiment testing, SKlearn Logistic Regression models, Crypto-Marketcap API, and Plotly for Data Visualizations. **This Code is best run via Google Colab, as all relevant files, information, and installations are done directly in our file interface.**

**Files Included**

twitter_crypto_sentiment.ipynb - This file contains all the code for the project (**Use this format tot run and view results of represented code)**

logreg_train.ipynb - ML model training file 

logreg.pkl - binary of logreg colab which is called in our main crypto sentiment file

tfidf_transformer.pkl - binary of tfidf_transformer found in logreg_train ipynb  file to run in the main

train_data.csv - training data specifically used in this project

database - folder which includes tweet data specifically used for visualizations

Project Proposal.pdf - PDF file proposing deliverables to be worked on for this project

Progress Report.pdf - Covers everything done and what was being worked on during the half-way point on the project.

Include PNG files of visualizations to be looked at by themselves

# Background

To cover the 4 main components of our work we have a Twitter Interface, Logistical Regression and Y-hat representation via  Sklearn, Crypto Market Cap API to pull closing Data, and Data Visualizations using Plotly.

Starting with the Twitter Interface, we're making use of the Twitter API Tweepy to pull tweets over a specified period of time defined under the search terms to then import to csv files. In this layer we aren't conducting the Sentiment Analysis, but we are doing a layer of pre-processing before comparing the most recent tweets pulled to our logistic regression models.

In regards to our model development we have 3 steps: training, preprocessing, and Logisitcal Regression:

For training data, we acquired sentiment labeled tweet datasets from a number of sources, including Kaggle and GitHub. Two bitcoin specific sets were used, as well as one each for Ethereum, Ripple, Bitcoin Cash, and Litecoin. An additional stock tweet dataset was used to provide additional signal on general financial terms. All these sets were concatenated together to form our main training dataset comprising of ~170k sentiment labeled tweets. Training labels could be -1 (negative), 0 (neutral), and 1 (positive).

This tweet set was preprocessed to remove usernams, http links, stopwords, and capitalization. Then the full dataset was split into train (70%), validation (15%), and test (15%) sets. Using sklearn, we transformed these datasets into tf-idf format where the transformer was derived from just the train set in order to avoid info leakage. We used mixture of unigrams and bigrams as this was experimentally determined to perform better using a single n-gram representation. Binary bag-of-words was also explored but performed slightly worse than tf-idf. A max vocab of 4000 was used for the transformer.

Using Logistic Regression as our model, we employed 5-fold cross validation to search for the optimal C hyperparemeter used in LogReg. Using base array of C values and sklearn's GridSearchCV we used an iterative hyperparamter search that would incrementally scope in to the best value found in the array and stop when the previous and current best hyperparameter would result in a difference of F1 score under some threshold. The best C value we found for this metric and data representation was 90.0. Using this hyperparameter value, we then trained on the training set and predicted on our test set. Train F1 (micro) score was ~86% and test F1 was 84.7%. F1 was used as our metric as the dataset was imbalanced to favor neutral and positive labels. The label distribution of our Y_hat_test vector was roughly similar to the distribution of labels in the dataset. The trained model and tf-idf transformer were then pickeled to be used with our twitter interface.

Next we have our Crypto Market Cap API which is crucial for our Data Visualization: we were taking the coinmarketcapapi library and returning a dataframe to include top ranking crypto currencies as an identifier to which we use to compare closing price with our trained data.

For our Data Visualizations we made use of plotly to plot out twitter user interface data into Sentiment Polarity v Time, Volume over Time, and Closing Averages over Time, we also included a plot of Polarity in direct comparison with the closing price to see how well our model is at predicting market sentiment.

For more details on each below includes Source Code Documentation with snippets and visuals for better understanding.

# Code Documentation

**Quick Intro**

List of Packages needed before beginning (if running locally rather than via Colab **please use the Google colab for the best experience**): Tweepy,Textblob, csv, numpy, matplotlib, pandas, json, re, datetime, codecs, nltk, pickle, sklearn, glob, plotly, coinmarketcap and google.colab for easiest access to files from your own drive.

Below is the basic coverage of some of the most important bits of our code:

Starting with the Tweepy API and interface, we must establish a connection by having a Twitter profile, and creating a developer account with Developer Access to which we get the following keys and ensure we have a connection:

```
consumer_key = ""
consumer_secret = ""
access_token  = ""
access_token_secret = ""

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit = True)
```

Instantiation of all our search terms, the number of tweets to pull, where to save them as a csv, which columns we'd like to include to write to csv, as well as the time and search query parameters we will pass can be set as such:

```
# All search terms we can use
search_term_xrp = '#$XRP OR #XRP OR #xrp OR #Ripple'
search_term_btc = '#BTC OR #Btc OR #btc OR #bitcoin'
search_term_eth = '#ETH OR #Eth OR #eth OR #ethereum'
search_term_bnb = '#BNB OR #Binance OR #bnb OR #binance'
search_term_doge = '#DOGE OR #Doge OR #doge'
search_term_litecoin = '#LTC OR #litecoin OR #ltc'
search_term_bitcoincash = '#BitcoinCash OR #BCH OR #bitcoincash'

no_of_tweets = 20

# setting specified timeframe to pull tweets from 
start_date = '2022-12-05T02:11:02'
end_date = '2022-12-06T03:11:01'
tweets = tweepy.Cursor(api.search, q = search_term_xrp, lang = "en", since = start_date, until = end_date).items(no_of_tweets)

xrp_tweets = 'ripple_tweets.csv'
btc_tweets = "btc_tweets.csv"
eth_tweets = "eth_tweets.csv"
bnb_tweets = "bnb_tweets.csv"
doge_tweets = "doge_tweets.csv"
litecoin_tweets = "litecoin_tweets.csv"
bitcoincash_tweets = "bitcoincash_tweets.csv"


# only including column for original text and created at
COLS = ['created_at','original_text']
```

Next we have our functions to write_tweets and preprocess them before we can send them to the Logistic Regression model

```
# Function to write tweets to separate csv files depending on the search terms of each crypto currency
def write_tweets(search_term, file):
  df = pd.DataFrame(columns = COLS)
  tweets = tweepy.Cursor(api.search, q = search_term, lang = "en", since = start_date, until = end_date).items(no_of_tweets)
  csvFile = open(file, 'a')
  csvWriter = csv.writer(csvFile)
  df.to_csv(csvFile, mode = 'a', columns = COLS, encoding = "utf-8", index  = False)
  for tweet in tweets:
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
    print(tweet.created_at, tweet.text)
# conduct same level of preprocessing as done in logestic regression model remove capitalization, stopwords, punctuation, and twitter link
def process(text): 
  text = str(text)
  text = text.lower()
  text = re.sub(r'@[^\s]+', '', text) # remove username
  text = re.sub(r'http\S+', '', text, flags=re.MULTILINE) # remove http link
  nopunc = [char for char in text if char not in string.punctuation]
  nopunc = ''.join(nopunc)
  procList = [word for word in nopunc.split() if word not in stopwords.words('english')]
  return ' '.join(procList)

# Change parameters of search term and file type to whatever you intend to query for (commented out all other 4 currencies for testing purposes)

write_tweets(search_term_btc, btc_tweets)
write_tweets(search_term_eth, eth_tweets)
write_tweets(search_term_xrp, xrp_tweets)
write_tweets(search_term_bnb, bnb_tweets)
write_tweets(search_term_doge, doge_tweets)
write_tweets(search_term_litecoin, litecoin_tweets)
write_tweets(search_term_bitcoincash, bitcoincash_tweets)    
```
From here we can take our tweets to be read into the Logistic Regression Model once we read in all the logreg and tfidf_transformer binary files to drive to be given the tfidf transformer and y_hat values. (**full model details can be viewed in the logreg_train.ipynb file**)

```
drive.mount('/content/drive', force_remount=True)
!ls /content/drive/MyDrive/CS410Project  

# Load trained logreg model state
with open('/content/drive/MyDrive/CS410Project/logreg.pkl', 'rb') as f:
    logreg = pickle.load(f)
    
# with open('/content/drive/MyDrive/CS410Project/tfidf_transformer.pkl', 'rb') as f: 
#    tfidf_transformer = pickle.load(f)
# The above approach is running into a dependency issue with scipy due to the usage of a sparse matrix data struct. 
# Installing the library in the notebook doesn't seem to work so we recreate the transformer from the train data


# Load train dataset to recreate the tf-idf transformer
data['text'] = data['text_clean'] # data can be changed 2 cells above to ping any specific crypto currency where tweets are pulled for 
train_data = pd.read_csv('/content/drive/MyDrive/CS410Project/train_data.csv', low_memory=False)
max_vocab = 4000 # don't change
tfidf_transformer = TfidfVectorizer(max_features=max_vocab, ngram_range=(1,2)).fit(train_data['text'].values.astype('U'))

x = tfidf_transformer.transform(data['text'].values.astype('U'))
y_hat = y_hat_train = logreg.predict(x)
```

Data Visualization is then conducted on this data, specifically reading in tweets from the csv file and running a layer of prediction on them using the trained model.

```
def percentage(part, whole):
    return 100 * float(part)/float(whole)

# initializing dataframes
crypto_name = "bitcoin" # we can change this to have user input value instead 
df_noOfTweets = pd.DataFrame(columns=['date','positive', 'neutral', 'negative', 'total']) # for the barchart of no. of tweets
df_polarity_line = pd.DataFrame(columns=['date','polarity']) # for the polarity line chart
# Tweets about each crypto is taken from saved txt files with dates ranging from 2017-09-01 to 2017-11-30
# Tweets are collected by selenium automated script written in python
for f in glob.glob( '/content/drive/MyDrive/database/Tweets/'+ crypto_name +'/'+ crypto_name + '*.txt'):
  df_datawise = pd.read_csv(f, sep='\t', encoding='utf-8', parse_dates=True)
  polarity = 0.00
    # df_datawise = pd.read_csv(f, sep='\t', encoding='utf-8', parse_dates=True)
    # df_datawise = pd.read_csv(btc_tweets, low_memory= False)
  df_datawise['Tweets'] = df_datawise['Tweets'].apply(lambda row: process(row))
  x = tfidf_transformer.transform(df_datawise['Tweets'].values.astype('U'))
  y_hat =  logreg.predict(x)
  polarity = 0.00
  neutral = positive = negative = 0
  for idx,row in df_datawise.iterrows():
      tweet_polarity = y_hat[idx]
      polarity += y_hat[idx]  # adding up polarities to find the average later
      
      # adding reaction of how people are reacting to find average later
      if (tweet_polarity == 1): 
          #print(positive) 
          positive += 1
          #raise NotImplementedError
      else:
          #print(negative)
          negative += 1
          #raise NotImplementedError
  positive = percentage(positive, len(df_datawise.index))
  negative = percentage(negative, len(df_datawise.index))
  neutral = percentage(neutral, len(df_datawise.index))

  # below df format is for barchart with sentiments
  # df_temp_df_noOfTweets = pd.DataFrame({'date': df_datawise.iloc[0]['date'], 'positive': ((positive/100)*len(df_datawise.index)),'neutral': ((neutral/100)*len(df_datawise.index)), 'negative': ((negative/100)*len(df_datawise.index)), 'total': len(df_datawise.index)}, index=[0])
  # df_noOfTweets = df_barchart.append(df_temp_df_noOfTweets, sort=True, ignore_index=True)

  # below df format is for polarity_line with sentiments
  df_temp_polarity_line = pd.DataFrame({'date': df_datawise.iloc[0]['date'], 'polarity': polarity}, index=[0])
  df_polarity_line = df_polarity_line.append(df_temp_polarity_line, sort=True, ignore_index=True)
print(df_polarity_line)
    
table = ff.create_table(df_noOfTweets)
# plotly.offline.iplot(table, filename='df_noOfTweets')
# =================== #
# no_of_tweets TABLE  #
# =================== #
```

Before we visualize the data after initialization, we also run our Coin Market API to get the current closing price to compare on our visualizations

```
cmc = coinmarketcapapi.CoinMarketCapAPI('b18b870d-d87e-4130-8243-6220c170e475')
data_id_map = cmc.cryptocurrency_map()
#print(data_id_map.data)
pdf = pd.DataFrame(data_id_map.data, columns =['id','name','symbol'])
#pdf.set_index('symbol',inplace=True)
id = pdf[pdf['symbol'] == 'BTC'].id
print(pdf)

# ===================================== #
# Getting the closing price and volume  #
# ===================================== #
#url='https://coinmarketcap.com/currencies/'+ crypto_name +'/historical-data/?start=20170901&end=20171130'
#url="https://coinmarketcap.com/currencies/"+ crypto_name +"/historical-data/?start=20170901&end=20171130"

api_url= 'https://api.coinmarketcap.com/data-api/v3/cryptocurrency/historical?id=1&convertId=2781&timeStart=1504249200&timeEnd=1512162061'
r = requests.get(api_url)
data = []
for item in r.json()['data']['quotes']:
    close = item['quote']['close']
    volume =item['quote']['volume']
    date=item['quote']['timestamp']
    data.append([close,volume,date])

cols = ["close", "volume","date"]

df = pd.DataFrame(data, columns= cols) 
df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d"))
df_marketcap_sentiment_line = pd.merge(df_polarity_line, df, left_on='date', right_on='date')
# ======================================= #
# Polarity, Vol. and Closing Price TABLE  #
# ======================================= #
table = ff.create_table(df_marketcap_sentiment_line)
#plotly.offline.iplot(table, filename='df_marketcap_sentiment_line')
#print(df_marketcap_sentiment_line.sort_values(by=["date"]))
```

From here we chart our Bar graphs showing Sentiment, Volume, and Closing averages over Time

```
# ==================================================================== #
# Bar Graphs of Each Sentiment, Volume, and Closing Average over Time  #
# ==================================================================== #

# df_market_cap_sentiment_line name of the df we're working with that has all the information 
#  columns go as such: close, volume, date

# barchart1 = make_subplots(specs =[[{"secondary_y": True}]])
barchart1 = px.bar(
    data_frame = df_marketcap_sentiment_line,
    title = "Crypto Closing Averages over Time",
    x = "date",
    y = "close",
    opacity = 0.9, 
    orientation = "v",
    barmode = 'relative',
    text = 'close',
    color = 'close',
)

barchart2 = px.bar(
    data_frame = df_marketcap_sentiment_line,
    title = "Sentiment Polarity over Time",
    x = "date",
    y = "polarity",
    opacity = 0.9, 
    orientation = "v",
    barmode = 'relative',
    text = 'close',
    color = 'polarity',
)
pio.show(barchart2)

barchart3 = px.bar(
    data_frame = df_marketcap_sentiment_line,
    title = "Volume over Time",
    x = "date",
    y = "volume",
    opacity = 0.9,
    orientation = "v",
    barmode = 'relative',
    text = 'close',
    color = 'volume',
)
pio.show(barchart3)

# data1 = [barchart1, barchart2]
pio.show(barchart1)
```
Visualizations of above code (**Clicking on the image will allow for a full view of them):**

![image](https://user-images.githubusercontent.com/111824503/206600752-594afb65-79a6-4603-bfd7-2480da9e2086.png)

![image](https://user-images.githubusercontent.com/111824503/206600907-a5e9feea-6582-4107-8c1c-ab0160edf816.png)

![image](https://user-images.githubusercontent.com/111824503/206600946-59467dde-5c84-4d95-9161-0aab92926e54.png)


And Finally our Sentiment Plot in comparison to Closing Price to compare our trained model to what happens in the market in relation to the time period selected

```
# ============================================== #
# Plot Polarity, Price in USD(Close) and Volume  #
# ============================================== #

trace1 = go.Scatter(
    x=df_marketcap_sentiment_line['date'],
    y=df_marketcap_sentiment_line['close'],
    mode = 'markers',
    name='Closing Price'
)
trace2 = go.Scatter(
    x=df_marketcap_sentiment_line['date'],
    y=df_marketcap_sentiment_line['polarity'],
    mode = 'markers',
    name='Polarity',
    yaxis='y2'
)
data = [trace1, trace2]
layout = go.Layout(
    title='Sentiment Polarity Vs. Closing Price (in USD)',
    yaxis=dict(
        title='Closing Price.'
    ),
    yaxis2=dict(
        title='Polarity',
        titlefont=dict(
            color='rgb(148, 103, 189)'
        ),
        tickfont=dict(
            color='rgb(148, 103, 189)'
        ),
        overlaying='y',
        side='right'
    )
)
fig = go.Figure(data=data, layout=layout)

# fig.show()

plotly.offline.iplot(fig, filename='Sentiment_vs_Market.html')
```

![image](https://user-images.githubusercontent.com/111824503/206601142-cbd63594-11d0-4b70-86a3-8a0084722c74.png)

# Team Contributions

Each member did a great job contributing to this project, as we started off the project losing a member (going from 5 to 4), each person played a role in Twitter Interface - Data Visualizations (Guru / Raza), Model Training and Parameter Tuning (Gabriel), and Coinmarket cap closing API implementation (Nusrat) to reach a total of 20 hours worked on the project each. 

**Improvements**

With more time we would refactor our visualizations to work better with the live Twitter Data, as a time constraint we had to use a pre-set of data to visualize with our trained model rather than having this truly live. So as part of working to improve this tool we would definitely work to have our Twitter API call more historical data rather than pulling from up to a 7 day period. 

With more time we could also make our model more verbose and increase the currencies accuracy to be trained upon. We also believe if we could have another layer of pre-processing to remove bot related tweets that tend to seap in from the data pool our model accuracy will definitely increase. 



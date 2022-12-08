# CourseProject

**Quick Note: If you want to run this source code yourself, you will need to gain Elevated Access to the Tweepy API to make API calls in quick succession. The provided Colab Notebook and Demo will contain all source code already ran to show results of each respective code block (this is also to ease grading workflow).**

This project repo covers the CS410 Text Information Systems Final Project conducted by our 4 man team. The main idea of this repo is to provide code, documentation, and a demo for a Twitter Sentiment Analysis Tool based on Crypto market data using Python, Tweepy, TextBlob for initial sentiment testing, SKlearn Logistic Regression models, Crypto-Marketcap API, and Plotly for Data Visualizations. **This Code is best run via Google Colab, as all relevant files, information, and installations are done directly in our file interface.**

**Files Included**

twitter-interface.ipynb - This file contains all the code for the project (**Use this format tot run and view results of represented code)**

main.py - Main source code file to view all code conducted for this project in one uniform location

Project Proposal.pdf - PDF file proposing deliverables to be worked on for this project

Progress Report.pdf - Covers everything done and what was being worked on during the half-way point on the project.

Include PNG files of visualizations to be looked at by themselves

# Background

To cover the 4 main components of our work we have a Twitter Interface, Logistical Regression and Y-hat representation via  Sklearn, Crypto Market Cap API to pull closing Data, and Data Visualizations using Plotly.

Starting with the Twitter Interface, we're making use of the Twitter API Tweepy to pull tweets over a specified period of time defined under the search terms to then import to csv files. In this layer we aren't conducting the Sentiment Analysis, but we are doing a layer of pre-processing before comparing the most recent tweets pulled to our logistic regression models.

**talk about the ML Model Implementation here**

Next we have our Crypto Market Cap API which is crucial for our Data Visualization: we were taking the coinmarketcapapi library and returning a dataframe to include top ranking crypto currencies as an identifier to which we use to compare closing price with our trained data.

For our Data Visualizations we made use of plotly to plot out twitter user interface data into Sentiment Polarity v Time, Volume over Time, and Closing Averages over Time, we also included a plot of Polarity in direct comparison with the closing price to see how well our model is at predicting market sentiment.

For more details on each below includes Source Code Documentation with snippets and visuals for better understanding.

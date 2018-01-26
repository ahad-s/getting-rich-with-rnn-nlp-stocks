# GET RICH QUICK! 

# USE MACHINE LEARNING AND CODE SKILLS FOR FREE $$$ IN THESE 4 STEPS!

Top of the line stock predictor from 1995

(A web crawler for news articles + sentiment analysis model (for positivity of text) with a W2V model and a semi-supervised ANN, with article sentiment used as a feature for stock price regression through a (LSTM) RNN)

# Phase 1 (Python)
- Steal all the data (Web crawler for articles) [DONE]
- AKA:
  - A web crawler/scraper for getting (timestamped) Bloomberg and (non-timestamped, only for training the model) Businesswire articles using Yahoo archive, running 24/7
  - Downloading intraday level 1 data from Dukascopy for popular US indices, FX rates, oil and some others

# Phase 2 (Python + Scripting)
- Call Mr. Clean for the data (Clean & preprocess articles) [DONE]
- AKA:
  - Articles gathered were cleaned and preprocessed (ex. removal of stopwords, parsing to proper tokens, etc.) with NLTK

# Phase 3 (Python)
- Convert the stolen data into Skynet (Sentiment analysis on articles) [DONE]
- AKA:
  - Created and trained a word2vec (CBOW) model with the articles gathered using the "gensim" library 
  - Implemented a single-layer semi-supervised neural network for predicting sentiment of the gathered articles

# Phase 4 (C++)
- Utilize Skynet to win Wall Street [DONE]
- AKA:
  - Regression of stock prices and article sentiment with a LSTM RNN
  
# Phase 5 (????)
- Inject steroided up Skynet into the NYSE/TSX

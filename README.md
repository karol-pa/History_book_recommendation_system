# A history book recommendation app 

As another excersize- (or hobby-) project my aim was to create a user friendly book recommendation app for all fans of ancient history (such as myself).
The app uses a simple [count vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) for finding books that are most similar to your own selection of favorite books.
All book information in this app are taken from [Openlibrary](https://openlibrary.org/). 

Once you have installed all required packages

```
pip install -r requirements.txt
```

run the following command from the terminal to open the app in your default web browser:


```
streamlit run app.py
```

Or try the [cloud version](https://historybookrecommendationsystem.streamlit.app/)



How to use: 

1. Click on a country on the map.
2. Select your favorite books associated with the selected country.
3. Repeat with other countries.
4. Click on the "Get Recommendations" button.

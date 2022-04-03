# Yelp Hybrid Recommender System
A Hybrid Recommender System for the Yelp! dataset

I have developed a restaurant cascade hybrid recommender system that takes into account three main components:
	⋅⋅*  Content-based filtering (CBF): using tf-idf feature weighting metric followed by cosine similarity
	⋅⋅* Collaborative filtering: takes refined results from CBF, and uses kNN with SVD, estimates ratings for a particular user and produces a final list for output
	⋅⋅* Geographical context: Refines results so restaurants are only shown if they are within users' visited geographical regions

Packages to install:
In order to run, you must install certain packages. Run `pip install` followed by the following for each package:
```
pandas
nltk
numpy
surprise
sklearn
```

How to run:
Ensure all the data files, 'business.pkl', 'user.pkl', 'review.pkl', 'business_review.pkl' and 'covid.pkl' are in the same directory as main.py.
Open a terminal in this directory, and type `python main.py`
Follow the on-screen instructions.

Want to test with an existing user? Try being James - with user-id: "BGKBgJCk-qyFpTYq-orgiw"

Need some example restaurants? Try "Banzai Sushi"
Want to test a chain restaurant? Try "Domino's Pizza"

## Implementation references:
[Understanding word stemming and pre-processing](https://www.kdnuggets.com/2020/08/content-based-recommendation-system-word-embeddings.html)

[Information on hybrid recommender systems](https://www.math.uci.edu/icamp/courses/math77b/lecture_12w/pdfs/Chapter%2005%20-%20Hybrid%20recommendation%20approaches.pdf)

[Understanding content-based systems](https://medium.com/analytics-vidhya/content-based-recommender-systems-in-python-2b330e01eb80)

[For understanding of collaborative systems](https://towardsdatascience.com/various-implementations-of-collaborative-filtering-100385c6dfe0)

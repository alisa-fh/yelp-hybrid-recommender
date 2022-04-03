# PACKAGES - no pip install required

import re
import random
from collections import defaultdict
import string
from datetime import datetime

# PACKAGES -  pip install required

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

from rake_nltk import Rake

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


from surprise import Reader
from surprise import Dataset
from surprise import prediction_algorithms


# Load files from the filtered pickled files
def load_pickle():
    df_yelp_business = pd.read_pickle('business.pkl')
    df_yelp_review = pd.read_pickle('review.pkl')
    df_yelp_user = pd.read_pickle('user.pkl')
    df_business_review = pd.read_pickle('business_review.pkl')
    df_yelp_covid = pd.read_pickle('covid.pkl')

    return df_yelp_business, df_yelp_review, df_yelp_user, df_business_review, df_yelp_covid

# Clean text function to clean any new reviews entered by users
r = Rake()
stemmer = SnowballStemmer('english')
def clean_text(text):
    # Remove punctuation
    text = text.translate(string.punctuation)

    # Convert words to lower case and split them
    text = text.lower().split()

    # Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    text = [stemmer.stem(i) for i in text]
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    r.extract_keywords_from_text(text)
    key_words_dict_scores = r.get_word_degrees()
    text = list(key_words_dict_scores.keys())

    return text

def process(df, df_yelp_user, df_yelp_business):
    df['date']  = pd.to_datetime(df['date'])
    df['week_day'] = df['date'].dt.weekday
    df['month'] = df['date'].dt.month
    df['hour'] = df['date'].dt.hour
    df = df.merge(df_yelp_user, on = 'user_id')
    df = df.merge(df_yelp_business, on = 'business_id')
    rename_dict = {'business_longitude': 'longitude', 'business_latitude': 'latitude',
              'business_state':'state','business_city':'city', 'business_address': 'address'}
    df = df.rename(columns = rename_dict)
    df = df.rename(columns={"stars_x": "rating", "stars_y": "stars"})
    return df

# Get the top n from a list of predictions
def get_top_n(predictions, n = 10):
    top_n = defaultdict(list)
    for (uid, iid, true_r, est, _) in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

# Predict user ratings using collaborative filtering
def predict_user_rating(algo, given_user, df_relevant_businesses, df_reviews, states_filter):
    user_predictions = []
    # Businesses in the user's state
    if len(states_filter) != 0:
        df_relevant_businesses = df_relevant_businesses.loc[df_relevant_businesses['state'].isin(states_filter)]
    business_id_unique = df_relevant_businesses.business_id.unique()
    for row in business_id_unique:
        user_predictions.append(algo.predict(given_user, row))
    return user_predictions

# Checks if the recommended places have been reviewed by the user and returns list of business_ids that have been
def check_if_reviewed(given_user, recommendations, df_reviews):
    business_list = []
    for business, _ in recommendations[given_user]:
        business_list.append(business)
    filtered = df_reviews.loc[(df_reviews['user_id'] == given_user) & (df_reviews['business_id'].isin(business_list))]
    visited = filtered["business_id"].values.tolist()
    return visited


# Converts business id list or tuples to their names for output
def business_id_to_name(business_data, df_business, type):
    business_list = []
    df_business = df_business.set_index("business_id")
    # Given a list of tuples of businesses and their ratings, output the business names
    if type == "tuples":
        for business, _ in business_data:
            name = df_business.at[business, "name"]
            business_list.append(name)
    # Simply given a list of business ids
    elif type == "array":
        for business in business_data:
            name = df_business.at[business, "name"]
            business_list.append(name)
    df_business = df_business.reset_index().rename(columns={df_business.index.name: "business_id"})
    return business_list


# Collaborative filtering using SVD
def collaborative(df_yelp_business, df_yelp_review, df_yelp_user, df_business_review, given_user):
    if given_user in df_yelp_review.user_id.values:
        visited_business_df = df_yelp_review.loc[df_yelp_review['user_id'] == given_user]
        visited_business_list = visited_business_df["business_id"].tolist()
        visited_business_df = df_yelp_business.loc[df_yelp_business['business_id'].isin(visited_business_list)]
        user_states = list(set(visited_business_df["state"].tolist()))
    else:
        user_states = []
    user_counts = df_yelp_review["user_id"].value_counts()

    # Limiting users to active users and their reviews:
    active_users = user_counts.loc[user_counts >= 3].index.tolist()
    df_yelp_review_active = df_yelp_review.loc[df_yelp_review.user_id.isin(active_users)]


    # Train - Test split
    SAMPLING_RATE = 1/5
    # Get list of unique user ids from active reviewers
    user_id_unique = df_yelp_review_active.user_id.unique()

    # Sample the users
    user_id_sample = pd.DataFrame(user_id_unique, columns=['unique_user_id']) \
                        .sample(frac= SAMPLING_RATE, replace=False, random_state=1)

    # Sample the reviews
    ratings_sample = df_yelp_review_active.merge(user_id_sample, left_on='user_id', right_on='unique_user_id') \
                        .drop(['unique_user_id'], axis=1)


    # hold out last review
    ratings_user_date = ratings_sample.loc[:, ['user_id', 'date']]
    ratings_user_date.date = pd.to_datetime(ratings_user_date.date)
    index_holdout = ratings_user_date.groupby(['user_id'], sort=False)['date'].transform(max) == ratings_user_date[
        'date']
    ratings_holdout_ = ratings_sample[index_holdout]
    ratings_traincv_ = ratings_sample[~index_holdout]

    ratings_user_date = ratings_traincv_.loc[:, ['user_id', 'date']]
    index_holdout = ratings_user_date.groupby(['user_id'], sort=False)['date'].transform(max) == ratings_user_date[
        'date']
    ratings_cv_ = ratings_traincv_[index_holdout]
    ratings_train_ = ratings_traincv_[~index_holdout]


    ratings_train = process(ratings_train_.copy(), df_yelp_user, df_yelp_business)
    ratings_test = process(ratings_holdout_.copy(), df_yelp_user, df_yelp_business)
    ratings_val = process(ratings_cv_.copy(), df_yelp_user, df_yelp_business)

    # remove observations that may cause cold-start problem, which breaks the model.
    ratings_test = ratings_test.loc[ratings_test.business_id.isin(ratings_train.business_id)]
    ratings_val = ratings_val.loc[ratings_val.business_id.isin(ratings_train.business_id)]

    trainset = ratings_train.reindex(columns = ['user_id', 'business_id', 'rating'])
    trainset.columns = ['userID', 'itemID', 'rating']
    valset = ratings_val.reindex(columns = ['user_id', 'business_id', 'rating'])
    valset.columns = ['userID', 'itemID', 'rating']
    testset = ratings_test.reindex(columns = ['user_id', 'business_id', 'rating'])
    testset.columns = ['userID', 'itemID', 'rating']

    reader = Reader(rating_scale=(0.0, 5.0))
    train_data = Dataset.load_from_df(trainset[['userID', 'itemID', 'rating']], reader)
    val_data = Dataset.load_from_df(valset[['userID', 'itemID', 'rating']], reader)
    test_data = Dataset.load_from_df(testset[['userID', 'itemID', 'rating']], reader)

    train_sr = train_data.build_full_trainset()
    val_sr_before = val_data.build_full_trainset()
    # val_sr = val_sr_before.build_testset()
    test_sr_before = test_data.build_full_trainset()
    # test_sr = test_sr_before.build_testset()

    # Uncomment to determine the best parameters for SVD
    # n_epochs = [10, 20, 30]  # the number of iterations
    # lr_all = [0.001, 0.003, 0.005]  # the learning rate for all parameters
    # reg_all = [0.02, 0.05, 0.1, 0.4, 0.5]  # the regularization term for all parameters
    #
    # RMSE_tune = {}
    # for n in n_epochs:
    #     for l in lr_all:
    #         for r in reg_all:
    #             print('Fitting n: {0}, l: {1}, r: {2}'.format(n, l, r))
    #             algo = prediction_algorithms.matrix_factorization.SVD(n_epochs=n, lr_all=l, reg_all=r)
    #             algo.fit(train_sr)
    #             predictions = algo.test(val_sr)
    #             RMSE_tune[n, l, r] = accuracy.rmse(predictions)
    import operator
    # minValues = min(RMSE_tune.items(), key=operator.itemgetter(1))[0]
    minValues = (30, 0.005, 0.1)

    # train and test on the optimal parameter
    algo_real = prediction_algorithms.matrix_factorization.SVD(n_epochs=minValues[0], lr_all=minValues[1], reg_all=minValues[2])
    algo_real.fit(train_sr)
    # predictions = algo_real.test(test_sr)
    # print(predictions)
    # accuracy.rmse(predictions)

    # TESTING:
    user_predictions = predict_user_rating(algo = algo_real, given_user=given_user, df_relevant_businesses=df_yelp_business, df_reviews=df_yelp_review_active, states_filter = user_states)
    top_n = get_top_n(user_predictions, 10)

    # Top ten business names in a list
    recommended_names = business_id_to_name(top_n[given_user], df_yelp_business, "tuples")
    print("We think you might like:")
    for i in range(4):
        print("\t", recommended_names[i])
    else:
        print("Some of the recommendations have been visited previously")

    # Finding the names of the businesses in reviewed_recommendations (i.e. names of recommended businesses that the
    # user has previously reviewed
    reviewed_recommendations = check_if_reviewed(given_user=given_user, recommendations=top_n,
                                                 df_reviews=df_yelp_review)
    filtered = df_yelp_business.loc[df_yelp_business['business_id'].isin(reviewed_recommendations)]
    visited_names = filtered["name"].values.tolist() # List of recommended and reviewed business names

    # List of business names that have been recommended
    recommended_names = business_id_to_name(top_n[given_user], df_yelp_business, "tuples")
    count = 0

    # Print business that are recommended and haven't been visited
    print("We think you might like:")
    for i in range(len(recommended_names)):
        if recommended_names[i] not in visited_names:
            print("\t", recommended_names[i])
            count += 1
        if count == 3:
            break
s
# kNN content recommender using cosine similarity
def content_recommender(df_business_review, df_business, given_business_id, relevant_states):
    df_business = df_business.set_index("business_id")
    # Filter business dataframe to businesses in the relevant state
    df_business = df_business.loc[df_business['state'].isin(relevant_states)]

    # Produce a list of the ids of relevant businesses
    relevant_businesses = list(df_business.index.values)

    # Dataframe of business and reviews for only relevant businesses and remove business_id as the index
    df_business_review = df_business_review.loc[df_business_review.index.isin(relevant_businesses)]
    df_business_review = df_business_review.reset_index().rename(columns={df_business_review.index.name: "business_id"})

    tfidf = TfidfVectorizer(stop_words= 'english')
    df_business_review['bag_of_words'] = df_business_review['bag_of_words'].fillna('')

    # Construct the required TF-IDF matrix by applying the fit_transform method on the overview feature
    overview_matrix = tfidf.fit_transform(df_business_review["bag_of_words"])
    # Output the shape of tfidf_matrix

    similarity_matrix = linear_kernel(overview_matrix, overview_matrix)

    # Mapping of index  and the business_id from the business_review dataframe
    mapping = pd.Series(df_business_review.index, index= df_business_review["business_id"])

    # Find the business index for the given business
    business_index = mapping[given_business_id]
    # Get similarity values with other businesses
    # Similarity_score is the list of index and similarity matrix
    similarity_score = list(enumerate(similarity_matrix[business_index]))
    # Sort in descending order the similarity score of business inputted with all the other businesses
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    # Get the scores of the 100 most similar businesses. Ignore the first business.
    similarity_score = similarity_score[1:100]
    # Return business names using the mapping series
    business_indices = [i[0] for i in similarity_score]
    recommended_ids = [df_business_review["business_id"][i] for i in business_indices]

    return recommended_ids

# Get the users preferences for better outputs
def check_preferences(user_id, df_user, df_business, df_covid, rec_tuples):
    if "user_id" in df_user.columns:
        df_user = df_user.set_index("user_id")
    if "business_id" in df_covid.columns:
        df_covid = df_covid.set_index("business_id")
    all_pref = ["-"] * len(rec_tuples)
    all_cov = ["-"] * len(rec_tuples)
    df_business = df_business.set_index("business_id")
    user_pref = df_user.at[user_id, "preferences"]
    i = 0
    if isinstance(user_pref, list):
        for business, _ in rec_tuples:
            att = []
            attributes = (df_business.at[business, "attributes"])
            if "1" in user_pref:
                if "GoodForKids" in attributes:
                    if attributes["GoodForKids"] == "True":
                        att.append("Good for kids")
            if "2" in user_pref:
                if "WheelchairAccessible" in attributes:
                    if attributes["WheelchairAccessible"] == "True":
                        att.append("Wheelchair accessible")
            if "3" in user_pref:
                if "RestaurantsTakeOut" in attributes:
                    if attributes["RestaurantsTakeOut"] == "True":
                        att.append("Takeout available")
            if len(att) != 0:
                all_pref[i] = att

            if df_covid.at[business, "delivery_or_takeout"] == "TRUE":
                all_cov[i] = "Currently delivering"
            elif df_covid.at[business, "delivery_or_takeout"] == "FAlSE":
                all_cov[i] = "Delivery currently unavailable"
            i += 1

    return all_pref, all_cov


# Hybrid of CF with CBF
def hybrid(user_id, business_id, df_yelp_business, df_yelp_review, df_yelp_user, df_business_review, df_yelp_covid):
    # Name of given business id
    given_business_name = business_id_to_name([business_id], df_yelp_business, "array")
    # Find business state
    if "business_id" in df_yelp_business.columns:
        df_yelp_business = df_yelp_business.set_index(["business_id"])
    business_state = df_yelp_business.at[business_id, "state"]
    df_yelp_business = df_yelp_business.reset_index().rename(columns={df_yelp_business.index.name: 'business_id'})
    # Find user states
    if user_id in df_yelp_review.user_id.values:
        visited_business_df = df_yelp_review.loc[df_yelp_review['user_id'] == user_id]
        visited_business_list = visited_business_df["business_id"].tolist()
        visited_business_df = df_yelp_business.loc[df_yelp_business['business_id'].isin(visited_business_list)]
        user_states = list(set(visited_business_df["state"].tolist() + [business_state] ))

    else:
        user_states = [business_state]

    # Find similar restaurants ids using content based
    content_results = content_recommender(df_business_review, df_yelp_business, business_id, user_states)

    df_recommended_business = df_yelp_business.loc[df_yelp_business['business_id'].isin(content_results)][["business_id","name", "state", "stars"]]

    user_counts = df_yelp_review["user_id"].value_counts()

    # Limiting users to active users and their reviews:
    active_users = user_counts.loc[user_counts >= 3].index.tolist()
    df_yelp_review_active = df_yelp_review.loc[df_yelp_review.user_id.isin(active_users)]

    # Train - Test split
    SAMPLING_RATE = 1/5
    # Get list of unique user ids from active reviewers
    user_id_unique = df_yelp_review_active.user_id.unique()

    # Sample the users
    user_id_sample = pd.DataFrame(user_id_unique, columns=['unique_user_id']) \
                        .sample(frac= SAMPLING_RATE, replace=False, random_state=1)

    # Sample the reviews
    ratings_sample = df_yelp_review_active.merge(user_id_sample, left_on='user_id', right_on='unique_user_id') \
                        .drop(['unique_user_id'], axis=1)


    # hold out last review
    ratings_user_date = ratings_sample.loc[:, ['user_id', 'date']]
    ratings_user_date.date = pd.to_datetime(ratings_user_date.date)
    index_holdout = ratings_user_date.groupby(['user_id'], sort=False)['date'].transform(max) == ratings_user_date[
        'date']
    ratings_holdout_ = ratings_sample[index_holdout]
    ratings_traincv_ = ratings_sample[~index_holdout]

    ratings_user_date = ratings_traincv_.loc[:, ['user_id', 'date']]
    index_holdout = ratings_user_date.groupby(['user_id'], sort=False)['date'].transform(max) == ratings_user_date[
        'date']
    ratings_cv_ = ratings_traincv_[index_holdout]
    ratings_train_ = ratings_traincv_[~index_holdout]

    ratings_train = process(ratings_train_.copy(), df_yelp_user, df_yelp_business)
    ratings_test = process(ratings_holdout_.copy(), df_yelp_user, df_yelp_business)
    ratings_val = process(ratings_cv_.copy(), df_yelp_user, df_yelp_business)


    # remove observations that may cause cold-start problem, which breaks the model.
    ratings_test = ratings_test.loc[ratings_test.business_id.isin(ratings_train.business_id)]
    ratings_val = ratings_val.loc[ratings_val.business_id.isin(ratings_train.business_id)]

    trainset = ratings_train.reindex(columns = ['user_id', 'business_id', 'rating'])
    trainset.columns = ['userID', 'itemID', 'rating']
    valset = ratings_val.reindex(columns = ['user_id', 'business_id', 'rating'])
    valset.columns = ['userID', 'itemID', 'rating']
    testset = ratings_test.reindex(columns = ['user_id', 'business_id', 'rating'])
    testset.columns = ['userID', 'itemID', 'rating']

    reader = Reader(rating_scale=(0.0, 5.0))
    train_data = Dataset.load_from_df(trainset[['userID', 'itemID', 'rating']], reader)
    val_data = Dataset.load_from_df(valset[['userID', 'itemID', 'rating']], reader)
    test_data = Dataset.load_from_df(testset[['userID', 'itemID', 'rating']], reader)

    train_sr = train_data.build_full_trainset()
    val_sr_before = val_data.build_full_trainset()
    # val_sr = val_sr_before.build_testset()
    test_sr_before = test_data.build_full_trainset()
    # test_sr = test_sr_before.build_testset()

    minValues = (30, 0.005, 0.1)

    algo_real = prediction_algorithms.matrix_factorization.SVD(n_epochs=minValues[0], lr_all=minValues[1], reg_all=minValues[2])
    algo_real.fit(train_sr)

    user_predictions = predict_user_rating(algo=algo_real, given_user=user_id, df_relevant_businesses=df_recommended_business,
                                            df_reviews=df_yelp_review, states_filter = [])
    top_n = get_top_n(user_predictions, 5)

    reviewed_recommendations = check_if_reviewed(given_user=user_id, recommendations=top_n,
                                                 df_reviews=df_yelp_review)

    print("-------- Because you visited", given_business_name[0],
          "we think you might like: --------")

    first_output = []
    recommended_names = business_id_to_name(top_n[user_id], df_yelp_business, "tuples")
    preferences, covid_data = check_preferences(user_id, df_yelp_user, df_yelp_business, df_yelp_covid, top_n[user_id])

    if len(reviewed_recommendations) == 0:
        # Top ten business names in a list
        print('\t{:<28s} {:<28s} {:<58s} {:<21s}'.format("RESTAURANT", "ESTIMATED RATING", "YOUR PREFERENCES", "COVID-19 CHANGES"))
        num_rec = max(5, len(recommended_names))
        for i in range(num_rec):
            print('\t{:<35s} {:<21s} {:<58s} {:<21s}'.format(recommended_names[i], str(round(top_n[user_id][i][1], 1)), ', '.join(preferences[i]), covid_data[i]))
            first_output += [recommended_names[i]]
        print("Why these? These recommendations have been made based on similar restaurants that users similar to you"
              " have enjoyed.\n")
    else:
        # Finding the names of the businesses in reviewed_recommendations (i.e. names of recommended businesses that the
        # user has previously reviewed
        filtered = df_yelp_business.loc[df_yelp_business['business_id'].isin(reviewed_recommendations)]
        visited_names = filtered["name"].values.tolist()  # List of recommended and reviewed business names
        # List of business names that have been recommended
        count = 0
        # Print business that are recommended and haven't been visited
        print('\t{:<28s} {:<28s}{:<58s}{:<21s}'.format("RESTAURANT", "ESTIMATED RATING", "YOUR PREFERENCES", "COVID-19 CHANGES"))
        for i in range(len(recommended_names)):
            if recommended_names[i] not in visited_names:
                print('\t{:<35s} {:<21s}{:<58s}{:<21s}'.format(recommended_names[i], str(round(top_n[user_id][i][1], 1)), ', '.join(preferences[i]), covid_data[i]))
                first_output += [recommended_names[i]]
                count += 1
            if count == 5:
                break

    # Don't repeat recommendations
    dont_recommend = first_output

    all_user_predictions = predict_user_rating(algo=algo_real, given_user=user_id,
                                           df_relevant_businesses=df_yelp_business, df_reviews=df_yelp_review_active, states_filter = user_states)
    top_n_collab = get_top_n(all_user_predictions, 20)
    if len(top_n_collab) > 0:
        reviewed_recommendations = check_if_reviewed(given_user=user_id, recommendations=top_n_collab,
                                                     df_reviews=df_yelp_review)
        recommended_names = business_id_to_name(top_n_collab[user_id], df_yelp_business, "tuples")
        preferences, covid_data = check_preferences(user_id, df_yelp_user, df_yelp_business, df_yelp_covid, top_n_collab[user_id])
        if len(reviewed_recommendations) > 0:
            filtered = df_yelp_business.loc[df_yelp_business['business_id'].isin(reviewed_recommendations)]
            # Add places that have been recommended so they aren't recommended again
            dont_recommend = first_output + filtered["name"].values.tolist()  # List of recommended and reviewed business names

        print("-------- Want to try something new? We think you might like these. --------" )
        print('\t{:<28s} {:<28s}{:<58s}{:<21s}'.format("RESTAURANT", "ESTIMATED RATING", "YOUR PREFERENCES", "COVID-19 CHANGES"))
        count = 0
        for i in range(len(recommended_names)):
            if recommended_names[i] not in dont_recommend:
                print('\t{:<35s} {:<21s}{:<58s}{:<21s}'.format(recommended_names[i], str(round(top_n_collab[user_id][i][1], 1)), ', '.join(preferences[i]), covid_data[i]))
                count += 1
            if count == 3:
                break
        print("Why these? Users similar to you liked these. They may not be related to", given_business_name[0], ".")


def welcome(df_user, df_business, df_review, df_business_review):
    df_user = df_user.set_index("user_id")
    df_business = df_business.set_index("business_id")
    print("Welcome to your Restaurant recommendation engine, sourcing your next favourite place to eat!")
    print("Are you a new or returning user?")

    user_option = input("\t1. Returning user \n\t2. New user\n")

    # Existing user
    if user_option == "1":
        user_id = input("Enter user id: ")
        # If user exists
        if user_id in df_user.index:
            user_name = df_user.at[user_id, "user_name"]
            print("Welcome back", user_name)
        else:
            print(user_id, " is not a user id in our system.")
            exit()

    # New user
    elif user_option == "2":
        print("Welcome! By joining our system you are agreeing to have your name and associated ID "
              "stored in our database for future login purposes and for the storage of your ratings and reviews."
              " Please create a user id:")
        while True:
            user_id = input("User ID: ")
            if user_id not in df_user.index:
                user_name = input("Name: ")
                print("Thanks for joining,", user_name)
                if "user_id" not in df_user.columns:
                    df_user = df_user.reset_index().rename(columns={df_user.index.name: "user_id"})
                # Add user to dataframe
                df_user = df_user.append({"user_id": user_id, "user_name": user_name}, ignore_index = True)
                # Update the pickled file to save permanently
                df_user.to_pickle('user.pkl')
                break
            else:
                print("Unfortunately, this ID is taken.")

    # Input is not 1 or 2
    else:
        print("Invalid option")
        exit()

    # Select a restaurant
    while True:
        business_name = input("Enter a restaurant you would like to rate: ")
        filtered_business = df_business.loc[df_business["name"] == business_name]
        if len(filtered_business.index) == 0:
            print("This restaurant is not in our records")
        else:
            break
    if len(filtered_business.index) == 1:
        business_id = filtered_business.index.values[0]
    else:
        print("Which ", business_name, "do you mean?")
        i = 1
        for _, row in filtered_business.iterrows():
            print(i, ". ", row["address"], row["state"], row["postal_code"])
            i += 1
        while True:
            user_option = input("Enter a number: ")
            if not user_option.isdigit():
                print("That is not a valid option")
            elif 0 < int(user_option) <= len(filtered_business.index):
                business_id = filtered_business.index[int(user_option) - 1]
                break
            else:
                print("That is not a valid option")


    print("Note: By submitting rating and review information, you agree for the information to be stored securely for"
          " future access and improved prediction purposes.")
    while True:
        rating_input = input("How many stars do you rate this restaurant (1-5)? ")
        if not rating_input.isdigit():
            print("That is not a valid option")
        elif 1 <= int(rating_input) <= 5:
            break
        else:
            print("That is not a valid option")

    review_id = ''.join(random.choice(string.ascii_letters) for i in range(22))
    while True:
        print("Would you like to write a review?")
        review_option = input("1. Yes\n2. No\n")
        if review_option == "1":
            review_input = input("Tell us about your experience:\n")
            review_input = clean_text(review_input)
            print("Thank you. Please wait as these are saved.")
            df_review = df_review.append({"review_id": review_id, "user_id": user_id, "business_id":
                business_id, "rating": rating_input, "text": review_input,
                "date": datetime.now().strftime("%d-%m-%Y %H:%M:%S")}, ignore_index=True)
            df_yelp_review.to_pickle('review.pkl')
            words = df_business_review.at[business_id, "bag_of_words"]
            words += ' '.join(review_input) + ' '
            df_business_review.at[business_id, "bag_of_words"] = words
            df_business_review.to_pickle('business_review.pkl')
            break
        elif review_option == "2":
            print("Your rating has been recorded. Please wait as it is saved.")
            df_review = df_review.append({"review_id": review_id, "user_id": user_id, "business_id":
                business_id, "rating": rating_input, "date": datetime.now().strftime("%d-%m-%Y %H:%M:%S")},
                ignore_index=True)
            df_review.to_pickle('review.pkl')
            break
        else:
            print("Invalid option")
    print("Would you like to update your preferences?")
    while True:
        pref_option = input("1. Yes\n2. No\n")
        if pref_option == "1":
            print("Which options do you prefer? Input all the relevant numbers separated by whitespace.")
            pref_input = input("1. Good for kids\n2. Wheelchair accessible\n3. Has takeout\n")
            pref_input = pref_input.split(" ")
            if "user_id" in df_user.columns:
                df_user = df_user.set_index("user_id")
            df_user.at[user_id, "preferences"] = pref_input
            df_user = df_user.reset_index().rename(columns={df_user.index.name: "user_id"})
            df_user.to_pickle('user.pkl')
            break
        elif pref_option == "2":
            break
        else:
            print("That is not a valid option")
    print("Please wait...")

    df_business = df_business.reset_index().rename(columns={df_business.index.name: "business_id"})
    hybrid(user_id = user_id, business_id = business_id, df_yelp_business= df_business, df_yelp_review=df_review, df_yelp_user = df_user, df_business_review=df_business_review, df_yelp_covid= df_yelp_covid)


df_yelp_business, df_yelp_review, df_yelp_user, df_business_review, df_yelp_covid = load_pickle()


welcome(df_user=df_yelp_user, df_business= df_yelp_business, df_review = df_yelp_review, df_business_review= df_business_review)

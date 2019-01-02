import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.colors as clr

def get_file_paths():
	user_files = []
	user_files.append('/Users/manthan/Data/cresci-2017.csv/datasets_full.csv/genuine_accounts.csv/users.csv')

	social_bot_files = []
	social_bot_files.append('/Users/manthan/Data/cresci-2017.csv/datasets_full.csv/social_spambots_1.csv/users.csv')
	social_bot_files.append('/Users/manthan/Data/cresci-2017.csv/datasets_full.csv/social_spambots_2.csv/users.csv')
	social_bot_files.append('/Users/manthan/Data/cresci-2017.csv/datasets_full.csv/social_spambots_3.csv/users.csv')

	spam_bot_files = []
	spam_bot_files.append('/Users/manthan/Data/cresci-2017.csv/datasets_full.csv/traditional_spambots_1.csv/users.csv')
	spam_bot_files.append('/Users/manthan/Data/cresci-2017.csv/datasets_full.csv/traditional_spambots_2.csv/users.csv')
	spam_bot_files.append('/Users/manthan/Data/cresci-2017.csv/datasets_full.csv/traditional_spambots_3.csv/users.csv')
	spam_bot_files.append('/Users/manthan/Data/cresci-2017.csv/datasets_full.csv/traditional_spambots_4.csv/users.csv')

	return user_files, social_bot_files, spam_bot_files


def pre_process(user_files, social_bot_files, spam_bot_files):

	final_data = pd.DataFrame({'A' : []})

	for user_file in user_files:

		temp_data = pd.read_csv(user_file, encoding='latin1')
		temp_data = temp_data[['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count', 'lang', 'time_zone', 'default_profile', 'geo_enabled', 'profile_use_background_image', 'profile_text_color', 'profile_sidebar_border_color', 'profile_background_tile', 'profile_sidebar_fill_color', 'profile_background_color', 'profile_link_color', 'utc_offset']]

		temp_data['label'] = int(0)
		
		final_data = final_data.append(temp_data, sort=True)
	
	final_data = final_data.drop('A', axis=1)

	for social_bot_file in social_bot_files:

		temp_data = pd.read_csv(social_bot_file, encoding='latin1')
		temp_data = temp_data[['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count', 'lang', 'time_zone', 'default_profile', 'geo_enabled', 'profile_use_background_image', 'profile_text_color', 'profile_sidebar_border_color', 'profile_background_tile', 'profile_sidebar_fill_color', 'profile_background_color', 'profile_link_color', 'utc_offset']]

		temp_data['label'] = int(1)

		final_data = final_data.append(temp_data, sort=True)

	for spam_bot_file in spam_bot_files:

		temp_data = pd.read_csv(spam_bot_file, encoding='latin1')
		temp_data = temp_data[['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count', 'lang', 'time_zone', 'default_profile', 'geo_enabled', 'profile_use_background_image', 'profile_text_color', 'profile_sidebar_border_color', 'profile_background_tile', 'profile_sidebar_fill_color', 'profile_background_color', 'profile_link_color', 'utc_offset']]
		
		temp_data['label'] = int(2)
		
		final_data = final_data.append(temp_data, sort=True)

	return final_data


def dictionary_encodings(column, feature):

	encoded_column = []
	encoding_dict = {}
	index = 0

	for i, row in column.iterrows():
		if row[feature] not in encoding_dict:
			encoding_dict[row[feature]] = index
			index = index + 1

	for i, row in column.iterrows():
		encoded_column.append(encoding_dict[row[feature]])

	return np.array(encoded_column)

def post_process(data, dictionary_encoding_features):

	data = data.fillna(0)

	for feature in dictionary_encoding_features:
		column = dictionary_encodings(data[:][[feature]], feature)
		data[feature] = column

	return data


def generate_data():

	user_files, social_bot_files, spam_bot_files = get_file_paths()
	data = pre_process(user_files, spam_bot_files, spam_bot_files)
	data = post_process(data, ['lang','time_zone','profile_text_color','profile_sidebar_border_color','profile_sidebar_fill_color','profile_background_color','profile_link_color'])

	print(data)

	y, X = data['label'], data.drop('label', axis=1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
	
	return X_train, y_train, X_test, y_test

def main():

	X_train, y_train, X_test, y_test = generate_data()

	followers = X_train['followers_count']
	friends = X_train['friends_count']
	statuses_count = X_train['statuses_count']

	importance = followers / (friends + followers)

	y = np.zeros((len(statuses_count,)))

	cdict = {0.0: 'green', 1.0:'red', 2.0:'red'}
	y_train = y_train.values
	for g in np.unique(y_train):
		i = np.where(y_train == g)
		print(i)
		plt.scatter(X_train[[x for x in i]], y_train[[x for x in i]])
	
	plt.show()


main()


# "id", "593932392663912449"

# "text", "RT @morningJewshow: Speaking about Jews and comedy tonight at Temple Emanu-El in San Francisco. In other words, my High Holidays."

# "user_id", "593932168524533760"
# "retweet_count",
# "reply_count",
# "favorite_count",
# "favorited",
# "retweeted", NULL,
# "possibly_sensitive", "0",
# "num_hashtags", "0",
# "num_urls", "1",
# "num_mentions",
# "created_at", "Fri May 01 00:18:11 +0000 2015",



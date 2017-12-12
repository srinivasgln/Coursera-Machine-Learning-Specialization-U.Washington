
# coding: utf-8

# # Recommending Songs based on various Models
# 
# ### Loading the data

# In[1]:

import graphlab
song_data=graphlab.SFrame('song_data.gl/')
graphlab.canvas.set_target('ipynb')


# In[2]:

# Exploring the data

song_data.head()


# In[3]:

len(song_data)


# # Counting the number of Unique users in the data we just read

# In[4]:

users=song_data['user_id'].unique()
len(users)


# # Create test and training data

# In[5]:

train_data,test_data=song_data.random_split(0.8,seed=0)


# # Popularity based recommender

# In[6]:

popularity_model=graphlab.popularity_recommender.create(train_data,user_id='user_id',item_id='song')


# In[7]:

# Using the popularity based model to make some predictions
# Recommendations for the first person in Users..
popularity_model.recommend(users=[users[0]])


# In[8]:

# Recommendations for the second person in Users..
popularity_model.recommend(users=[users[1]])


# # Personalized recommender model
#         

# In[9]:

personalized_model=graphlab.item_similarity_recommender.create(train_data,user_id='user_id',
                                                              item_id='song')


# # Now making some song secommendations based on our personalized model..

# In[10]:

# Recommendations for first user on Users
personalized_model.recommend(users=[users[0]])


# In[11]:

# Recommendations for second user on Users
personalized_model.recommend(users=[users[1]])


# # Acquiring the similar items based on our personalized model..

# In[12]:

personalized_model.get_similar_items(['With Or Without You - U2'])


# In[13]:

personalized_model.get_similar_items(['Chan Chan (Live) - Buena Vista Social Club'])


# # Comparing various recommendation models using Precision- Recall Curve

# In[14]:

get_ipython().magic(u'matplotlib inline')
model_performance=graphlab.recommender.util.compare_models(test_data,[popularity_model,personalized_model],
                                                          user_sample=0.05)


# # Counting unique users: 
# The method .unique() can be used to select the unique elements in a column of data. In this question, you will compute the number of unique users who have listened to songs by various artists. For example, to find out the number of unique users who listened to songs by 'Kanye West', all you need to do is select the rows of the song data where the artist is 'Kanye West', and then count the number of unique entries in the ‘user_id’ column. Compute the number of unique users for each of these artists: 'Kanye West', 'Foo Fighters', 'Taylor Swift' and 'Lady GaGa'. Save these results to answer the quiz at the end.
# 

# In[15]:

# Number of unique users who listended to Kanye west..
len(song_data[song_data['artist']=='Kanye West']['user_id'].unique())


# In[16]:

# Number of unique users who listended to Foo Fighters..
len(song_data[song_data['artist']=='Foo Fighters']['user_id'].unique())


# In[17]:

# Number of unique users who listended to Taylor Swift..
len(song_data[song_data['artist']=='Taylor Swift']['user_id'].unique())


# In[18]:

# Number of unique users who listended to Taylor Swift..
len(song_data[song_data['artist']=='Lady GaGa']['user_id'].unique())


# ## Using groupby-aggregate to find the most popular and least popular artist: 
# 
# Each row of song_data contains the number of times a user listened to particular song by a particular artist. If we would like to know how many times any song by 'Kanye West' was listened to, we need to select all the rows where ‘artist’=='Kanye West' and sum the ‘listen_count’ column. If we would like to find the most popular artist, we would need to follow this procedure for each artist, which would be very slow. Instead, you will learn about a very important method .groupby()

# In[19]:

total_count=song_data.groupby(key_columns='artist', operations={'total_count': 
                                                                graphlab.aggregate.SUM('listen_count')})


# In[20]:

# Most Popular artists
total_count.sort(['total_count'],ascending=False)


# In[21]:

# Least Popular artists
total_count.sort(['total_count'],ascending=True)


# # Using groupby-aggregate to find the most recommended songs: 
# Now that we learned how to use .groupby() to compute aggregates for each value in a column, let’s use to find the song that is most recommended by the personalized_model model we learned in the iPython notebook above

# In[22]:


subset_test_users = test_data['user_id'].unique()[0:10000]
recom=personalized_model.recommend(subset_test_users,k=1)
recom.show()


# In[23]:

# Most played song 
song_played=recom.groupby(key_columns='song', operations={'total_count': 
                                              graphlab.aggregate.COUNT('listen_count')})

song_played.sort(['total_count'],ascending=False)


# In[26]:

# Least played song 
song_played.sort(['total_count'],ascending=True)


# In[ ]:




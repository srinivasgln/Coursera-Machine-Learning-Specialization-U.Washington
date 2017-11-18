
# coding: utf-8

# # Document Retrieval from Wikipedia based on similarity

# In[1]:

import graphlab


# # Loading Text data from Wikipedia abovt famous people

# In[2]:

people=graphlab.SFrame('people_wiki.gl/')
people.head()


# In[3]:

len(people)


# # Exploring the data in People and checking the information it contains..

# In[4]:

obama=people[people['name']=='Barack Obama']
obama


# In[5]:

obama['text']


# In[6]:

clooney=people[people['name']=='George Clooney']
clooney['text']


# # Obtaining the wordcount for the Obama article

# In[7]:

obama['word_count']=graphlab.text_analytics.count_words(obama['text'])
print(obama['word_count'])


# # Sorting the wordcount we found for the obama article
# We use .stack() for transforming the dictionary to a table

# In[8]:

obama_word_count_table=obama[obama['word_count']].stack('word_count',new_column_name=['word','count'])
obama_word_count_table.head()


# In[9]:

obama_word_count_table.sort('count',ascending=False)


# # Determining the tf-idf for the whole corpus i.e people SFrame
# ## Word count for the entire corpus

# In[10]:

people['word_count']=graphlab.text_analytics.count_words(people['text'])
people.head()


# ## Now we can proceed to find the tf-idf

# In[11]:

people['tfidf'] = graphlab.text_analytics.tf_idf(people['word_count'])
people.head()


# In[12]:

people['tfidf']


# # Examining the tf-idf for the obama article
# 
# we must read the obama again as we have added new columns to the SFrame people

# In[13]:

obama=people[people['name']=='Barack Obama']


# In[14]:

obama[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)


# # Manually Compare the distances between few people

# In[15]:

clinton = people[people['name'] == 'Bill Clinton']
beckham = people[people['name'] == 'David Beckham']


# # Is obama closer to clinton than to Beckham?
# 
# Comparing the similarity  between the wikipedia article to clinton's and Beckham's

# In[16]:

graphlab.distances.cosine(obama['tfidf'][0],clinton['tfidf'][0])
#(0) represents the zeroth row


# In[17]:

graphlab.distances.cosine(obama['tfidf'][0],beckham['tfidf'][0])


# ## Therefore, it's clear that Clinton article is closer in similarity to Obama than the Beckham

# # For retrieving the similar documents using nearest neighbour model
# 
# Building a nearest neighbour model..,
# 

# In[18]:

knn_model=graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name')


# ## Applying the nearest neighbours model for retrieval of similar articles..

# In[19]:

# Who is closer to obama in the whole Sfram People??

knn_model.query(obama)


# # Other examples of document retrieval
# 
# Who is closer in similarity to whom?

# In[20]:

swift=people[people['name']=='Taylor Swift']
knn_model.query(swift)


# In[21]:

jolie=people[people['name']=='Angelina Jolie']
knn_model.query(jolie)


# In[22]:

arnold=people[people['name']=='Arnold Schwarzenegger']
knn_model.query(arnold)


# # Assignment
# 
# Now you are ready! We are going do three tasks in this assignment. There are several results you need to gather along the way to enter into the quiz after this reading.
# 
# ## Compare top words according to word counts to TF-IDF: 
# 
# In the notebook we covered in the module, we explored two document representations: word counts and TF-IDF. Now, take a particular famous person, 'Elton John'. What are the 3 words in his articles with highest word counts? What are the 3 words in his articles with highest TF-IDF? These results illustrate why TF-IDF is useful for finding important words. Save these results to answer the quiz at the end.
# ## Measuring distance:
# 
# Elton John is a famous singer; let’s compute the distance between his article and those of two other famous singers. In this assignment, you will use the cosine distance, which one measure of similarity between vectors, similar to the one discussed in the lectures. You can compute this distance using the graphlab.distances.cosine function. What’s the cosine distance between the articles on ‘Elton John’ and ‘Victoria Beckham’? What’s the cosine distance between the articles on ‘Elton John’ and Paul McCartney’? Which one of the two is closest to Elton John? Does this result make sense to you? Save these results to answer the quiz at the end.
# 
# ## Building nearest neighbors models with different input features and setting the distance metric:
# 
# In the sample notebook, we built a nearest neighbors model for retrieving articles using TF-IDF as features and using the default setting in the construction of the nearest neighbors model. Now, you will build two nearest neighbors models:
# Using word counts as features
# Using TF-IDF as features
# In both of these models, we are going to set the distance function to cosine similarity. Here is how: when you call the function
# 
# 
# Now we are ready to use our model to retrieve documents. Use these two models to collect the following results:
# 
# What’s the most similar article, other than itself, to the one on ‘Elton John’ using word count features?
# What’s the most similar article, other than itself, to the one on ‘Elton John’ using TF-IDF features?
# What’s the most similar article, other than itself, to the one on ‘Victoria Beckham’ using word count features?
# What’s the most similar article, other than itself, to the one on ‘Victoria Beckham’ using TF-IDF features?
# Save these results to answer the quiz at the end.

# # For the person,  Elton John
# What are the 3 words in his articles with highest word counts? What are the 3 words in his articles with highest TF-IDF?

# In[23]:

# Using regular wordcount (TF)
ejohn=people[people['name']=='Elton John']
ejohn['word_count'] = graphlab.text_analytics.count_words(ejohn['text'])
ejohn_word_count_table = ejohn[['word_count']].stack('word_count', new_column_name = ['word','count'])
ejohn_word_count_table.sort('count',ascending=False)


# In[24]:

# Using tfidf
ejohn[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)


# # Measuring the similarity between the EJohn and other two famous singers 
# 
# What’s the cosine distance between the articles on ‘Elton John’ and ‘Victoria Beckham’? What’s the cosine distance between the articles on ‘Elton John’ and Paul McCartney’? Which one of the two is closest to Elton John? Does this result make sense to you? 

# In[25]:

#What’s the cosine distance between the articles on ‘Elton John’ and ‘Victoria Beckham’?
victoria = people[people['name'] == 'Victoria Beckham']
#victoria[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)
graphlab.distances.cosine(ejohn['tfidf'][0],victoria['tfidf'][0])


# In[26]:

#What’s the cosine distance between the articles on ‘Elton John’ and Paul McCartney’?
mccartney = people[people['name'] == 'Paul McCartney']
graphlab.distances.cosine(ejohn['tfidf'][0],mccartney['tfidf'][0])


# # Building nearest neighbors models with different input features and setting the distance metric: 

# ## Nearest neighbour model using Word Count 

# In[27]:

wc_model=graphlab.nearest_neighbors.create(people,distance='cosine',features=['word_count'],
                                           label='name')
wc_model


# # What’s the most similar article, other than itself, to the one on ‘Elton John’ using word count features?

# In[28]:

ejohn=people[people['name']=='Elton John']
wc_model.query(ejohn)


# # What’s the most similar article, other than itself, to the one on ‘Victoria Beckham’ using word count features?

# In[29]:

victoria=people[people['name']=='Victoria Beckham']
wc_model.query(victoria)


# # Nearest neighbour model using tf-idf
# What’s the most similar article, other than itself, to the one on ‘Elton John’ using TF-IDF features?
# 
# What’s the most similar article, other than itself, to the one on ‘Victoria Beckham’ using TF-IDF features?
# 

# In[30]:

tfidf_model=graphlab.nearest_neighbors.create(people,distance='cosine',features=['tfidf'],label='name')
tfidf_model


# In[31]:

tfidf_model.query(ejohn)


# In[32]:

tfidf_model.query(victoria)


# In[ ]:





# coding: utf-8

# # Analysing the product Sentiment

# In[1]:

import graphlab
#Reading the product review data amazon-baby into graphlab
products=graphlab.SFrame('amazon_baby.gl')


# ## Visualizing the products data ,

# In[2]:

products.head()


# In[3]:

#Building the word count vector for each review, in a seperate column..
products['word_count']=graphlab.text_analytics.count_words(products['review'])
products.head()


# In[5]:

graphlab.canvas.set_target('ipynb')
products['name'].show() # histogram


# In[6]:

#Exploring the most widely reviewed product, Vulli Sophie the Giraffe Teether
giraffe_reviews=products[products['name']=='Vulli Sophie the Giraffe Teether']
len(giraffe_reviews)


# In[7]:

#Viewing  the product, vulli sophie the giraffe, by the ratings,
giraffe_reviews['rating'].show(view='Categorical')


# In[11]:

# Viewing # of ratings of all products 
products['rating'].show(view='Categorical')


# # Building the Sentiment Classifier
# 
# First we have to define what's a positive and a negative sentiment

# In[12]:

# ignore all rows in products with 3 star review
products=products[products['rating']!=3]
# the positive sentiment for our model is the products with rating 4 or 5 stars
products['sentiment']=products['rating']>=4


# # Training our Sentiment Classifier model with training data

# In[13]:

train_data,test_data=products.random_split(0.8,seed=0)
print(products.shape)
print(train_data.shape)
print(166752*0.80)
sentiment_model=graphlab.logistic_classifier.create(train_data,target='sentiment',
                                                    features=['word_count'],validation_set=test_data)


# # Evaluating the sentiment model 

# In[15]:

sentiment_model.evaluate(test_data,metric='roc_curve')
# Roc_curve is to explore the false positive and false negative
sentiment_model.show(view='Evaluation')



# # Applying learned model to understand sentiment for Vulli sophie Giraffe:

# In[16]:

giraffe_reviews['predicted_sentiment']=sentiment_model.predict(giraffe_reviews,
                                                               output_type='probability')
giraffe_reviews.head()


# # Sorting the reviews based on the predicted sentiment and explore the data

# In[17]:

giraffe_reviews=giraffe_reviews.sort('predicted_sentiment',ascending=False)
giraffe_reviews.head()


# ## To see the first review of the above table:

# In[18]:

giraffe_reviews[0]['review']


# ## To see the second review of the above table:

# In[19]:

giraffe_reviews[1]['review']


# # Now visualizing the negative reviews on the table giraffe_review.
# ## i.e the last review of the giraffe_review table.

# In[20]:

giraffe_reviews[-1]['review']


# # Assignment

# # Question 1:
# Now you are ready! We are going do four tasks in this assignment. There are several results you need to gather along the way to enter into the quiz after this reading.
# 
# In the IPython notebook above, we used the word counts for all words in the reviews to train the sentiment classifier model. Now, we are going to follow a similar path, but only use this subset of the words:
# 
# selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']
# 
# Often, ML practitioners will throw out words they consider “unimportant” before training their model. This procedure can often be helpful in terms of accuracy. Here, we are going to throw out all words except for the very few above. Using so few words in our model will hurt our accuracy, but help us interpret what our classifier is doing.
# # Use .apply() to build a new feature with the counts for each of the selected_words.
# 
#  In the notebook above, we created a column ‘word_count’ with the word counts for each review. Our first task is to create a new column in the products SFrame with the counts for each selected_word above, and, in the process, we will see how the method .apply() can be used to create new columns in our data (our features) and how to use a Python function, which is an extremely useful concept to grasp!
# Our first goal is to create a column products[‘awesome’] where each row contains the number of times the word ‘awesome’ showed up in the review for the corresponding product, and 0 if the review didn’t show up. One way to do this is to look at the each row ‘word_count’ column and follow this logic:
# 
# If ‘awesome’ shows up in the word counts for a particular product (row of the products SFrame), then we know how often ‘awesome’ appeared in the review,
# if ‘awesome’ doesn’t appear in the word counts, then it didn’t appear in the review, and we should set the count for ‘awesome’ to 0 in this review.
# We could use a for loop to iterate this logic for each row of the products SFrame, but this approach would be really slow, because the SFrame is not optimized for this being accessed with a for loop. Instead, we will use the .apply() method to iterate the the logic above for each row of the products[‘word_count’] column (which, since it’s a single column, has type SArray). Read about using the .apply() method on an SArray here.
# 
# We are now ready to create our new columns:
# 
# First, you will use a Python function to define the logic above. You will write a function called awesome_count which takes in the word counts and returns the number of times ‘awesome’ appears in the reviews.
# A few tips:
# 
# i. Each entry of the ‘word_count’ column is of Python type dictionary.
# 
# Next, you will use .apply() to iterate awesome_count for each row of products[‘word_count’] and create a new column called ‘awesome’ with the resulting counts. Here is what that looks like:
# 
# products['awesome'] = products['word_count'].apply(awesome_count)
# 
# And you are done! Check the products SFrame and you should see the new column you just create.
# 
# Repeat this process for the other 11 words in selected_words. (Here, we described a simple procedure to obtain the counts for each selected_word. There are other more efficient ways of doing this, and we encourage you to explore this further.)
# Using the .sum() method on each of the new columns you created, answer the following questions: Out of the selected_words, which one is most used in the dataset? Which one is least used? Save these results to answer the quiz at the end.
# 
# 

# In[21]:

selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 
                  'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']
def SW_count(dict_wordcount):
    if Current_SW in dict_wordcount:
        return dict_wordcount[Current_SW] 
    else:
        return 0
for SW in selected_words:
    Current_SW=SW
    products[SW] = products['word_count'].apply(SW_count)
    
sum_SelectedWords={}
for SW in selected_words:
    sum_SelectedWords[SW]=products[SW].sum()
#products.head()

print(sum_SelectedWords)

count=0
for m,n in sum_SelectedWords.items():
    count=count+1
    if count==1:
        bigCount=0
        smallCount=n
    if n>bigCount:
        bword=m
        bigCount=n
    if n<smallCount:
        smallCount=n
        sword=m
print('The largest key is',bword,'and its count is',bigCount)
print('the smallest key is',sword,'and its count is',smallCount)



# # Question 2:
# Create a new sentiment analysis model using only the selected_words as features: In the IPython Notebook above, we used word counts for all words as features for our sentiment classifier. Now, you are just going to use the selected_words.
# 
# Train a logistic regression classifier (use graphlab.logistic_classifier.create) using just the selected_words. Hint: you can use this parameter in the .create() call to specify the features used to be exactly the new columns you just created.
# 
# Call your new model: selected_words_model.
# 
# You will now examine the weights the learned classifier assigned to each of the 11 words in selected_words and gain intuition as to what the ML algorithm did for your data using these features. In GraphLab Create, a learned model, such as the selected_words_model, has a field 'coefficients', which lets you look at the learned coefficients. You can access it by using
# 
# selected_words_model['coefficients']
# 
# The result has a column called ‘value’, which contains the weight learned for each feature.
# 
# Using this approach, sort the learned coefficients according to the ‘value’ column using .sort(). Out of the 11 words in selected_words, which one got the most positive weight? Which one got the most negative weight? Do these values make sense for you? Save these results to answer the quiz at the end.

# In[22]:

# Training the new Selected_words_model using only selected words
train_data,test_data = products.random_split(.8, seed=0)
selected_words_model=graphlab.logistic_classifier.create(train_data,target='sentiment'
                                                        ,features=selected_words,validation_set=test_data)
# sorting Value to get most positive and most negative number
most_positive=selected_words_model['coefficients'].sort('value',ascending=False).head(1)
most_negative=selected_words_model['coefficients'].sort('value',ascending=True).head(1)
print('The most positive word is ',most_positive)
print('The most negative word is ',most_negative)


# # Question 3:
# What is the accuracy of the selected_words_model on the test_data? What was the accuracy of the sentiment_model that we learned using all the word counts in the IPython Notebook above from the lectures? What is the accuracy majority class classifier on this task? How do you compare the different learned models with the baseline approach where we are just predicting the majority class? Save these results to answer the quiz at the end.
# 
# Hint: we discussed the majority class classifier in lecture, which simply predicts that every data point is from the most common class. This is baseline is something we definitely want to beat with models we learn from data.

# In[30]:

# Evaluating the selected_words_model
selected_words_model.evaluate(test_data,metric='roc_curve')
# Roc_curve is to explore the false positive and false negative
selected_words_model.show(view='Evaluation')


# Accuracy of majority class cassifier
'''Ratings of 4 and 5 stars are considered as positive sentiment (or 1) 
 while rating 1 and 2 stars are treated as negative sentiment ( or sentiment 0)'''
totalRate=test_data.num_rows()
posR=test_data[(test_data['rating'] == 4)|(test_data['rating'] == 5) ].num_rows()
negR=test_data[(test_data['rating'] == 1)|(test_data['rating'] == 2) ].num_rows()
print('the total number of positive reviews are',posR,'the total number of negative reviews are'
     ,negR,'total number of reviews are ',totalRate)
majority_class_clsf=float(posR)/float(totalRate)
print('the accuracy of the majority class classifier is',majority_class_clsf)


# # Question 4:
# 
# Interpreting the difference in performance between the models: To understand why the model with all word counts performs better than the one with only the selected_words, we will now examine the reviews for a particular product.
# 
# We will investigate a product named ‘Baby Trend Diaper Champ’. (This is a trash can for soiled baby diapers, which keeps the smell contained.)
# Just like we did for the reviews for the giraffe toy in the IPython Notebook in the lecture video, before we start our analysis you should select all reviews where the product name is ‘Baby Trend Diaper Champ’. Let’s call this table diaper_champ_reviews.
# Again, just as in the video, use the sentiment_model to predict the sentiment of each review in diaper_champ_reviews and sort the results according to their ‘predicted_sentiment’.
# What is the ‘predicted_sentiment’ for the most positive review for ‘Baby Trend Diaper Champ’ according to the sentiment_model from the IPython Notebook from lecture? Save this result to answer the quiz at the end.
# Now use the selected_words_model you learned using just the selected_words to predict the sentiment most positive review you found above. Hint: if you sorted the diaper_champ_reviews in descending order (from most positive to most negative), this command will be helpful to make the prediction you need:
# 
# selected_words_model.predict(diaper_champ_reviews[0:1], output_type='probability')
# 
# Why is the predicted_sentiment for the most positive review found using the model with all word counts (sentiment_model) much more positive than the one using only the selected_words (selected_words_model)? Hint: examine the text of this review, the extracted word counts for all words, and the word counts for each of the selected_words, and you will see what each model used to make its prediction. Save this result to answer the quiz at the end.

# In[27]:

# Applying the models we built to product Baby Trend Diaper Champ


diaper_champ_reviews=products[products['name']=='Baby Trend Diaper Champ']

# 1) Using sentiment_model
diaper_champ_reviews['predicted_sentiment']=sentiment_model.predict(diaper_champ_reviews,
                                                                    output_type='probability')
diaper_champ_reviews=diaper_champ_reviews.sort('predicted_sentiment',ascending=False)
diaper_champ_reviews.head()




# In[28]:

# 2) USing Selected_words_model
selected_words_model.predict(diaper_champ_reviews[0:1], output_type='probability')


# In[ ]:




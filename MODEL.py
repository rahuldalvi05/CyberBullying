#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import nltk

from flask import Flask, render_template, request, redirect
# In[2]:
import pymysql
app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))



raw=pd.read_csv("cleanprojectdataset.csv")
raw['Tweet']=raw.Tweet.apply(lambda x:x.replace('.',''))
raw.head()

#raw.shape[0]


# In[3]:


import nltk
import string 
import re

nltk.download('stopwords')
nltk.download('wordnet')

#Create lists for tweets and label
Tweet = []
Labels = []
wpt = nltk.WordPunctTokenizer()
wordnet_lemmatizer = nltk.WordNetLemmatizer()


for row in raw["Tweet"]:
    #tokenize words
    words = wpt.tokenize(row)
    #remove punctuations
    clean_words = [word.lower() for word in words if word not in set(string.punctuation)]
    #remove stop words
    english_stops = nltk.corpus.stopwords.words('english')
    characters_to_remove = ["''",'``',"rt","https","’","“","”","\u200b","--","n't","'s","...","//t.c" ]
    clean_words = [word for word in clean_words if word not in english_stops]
    clean_words = [word for word in clean_words if word not in set(characters_to_remove)]
    #Lematise words
    lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]
    #print(lemma_list)
    Tweet.append(lemma_list)

    
    
for label in raw["Class"]:
    Labels.append(label)
    
c=[]
    
for x in Tweet:
    s=''
    for y in x:
            s=s+' '+y
    s = re.sub('[^A-Za-z0-9" "]+', '', s)
    s=s.lstrip()
    c.append(s)

#print(c)

from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
tv_matrix = tv.fit_transform(c)
tv_matrix = tv_matrix.toarray()

vocab = tv.get_feature_names()

lol=pd.DataFrame(np.round(tv_matrix, 2), columns=vocab)


# In[14]:





# the final preprocessing step is to divide data into training and test sets
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(tv_matrix, Labels, test_size = 0.45,random_state=0)

# TYPE your Code here
# Training the Algorithm. Here we would use simple SVM , i.e linear SVM
#simple SVM

from sklearn.svm import SVC
svclassifier=SVC(kernel='linear')



# classifying linear data


# kernel can take many values like
# Gaussian, polynomial, sigmoid, or computable kernel
# fit the model over data

svclassifier.fit(X_train,y_train)


# Making Predictions

y_pred=svclassifier.predict(X_test)

#print(y_pred)


# Evaluating the Algorithm

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

#print(confusion_matrix(y_test,y_pred))

#print(classification_report(y_test,y_pred))

#print(accuracy_score(y_test,y_pred))

# Remember : for evaluating classification-based ML algo use  
# confusion_matrix, classification_report and accuracy_score.
# And for evaluating regression-based ML Algo use Mean Squared Error(MSE), ...


# In[15]:


from sklearn.naive_bayes import GaussianNB


model = GaussianNB()


model.fit(X_train,y_train)



predicted=model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

#print(confusion_matrix(y_test,y_pred))

#print(classification_report(y_test,y_pred))

#print(accuracy_score(y_test,y_pred))


# In[16]:


results = model.predict_proba(X_test)[1]

#results


# In[17]:


svm = SVC(probability=True)
from sklearn.svm import SVC
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)
LABEL=svm.predict(X_test)
#print(LABEL[1:10])
output_proba = svm.predict_proba(X_test)
#print(output_proba)


# In[18]:


import sys
import tweepy


# credentials  --> put your credentials here
consumer_key = "OY4JG8cJFj3tjV6YJublPnsp8"
consumer_secret = "Ka44kCPfD1P8yV5XVk0x71AM012iDOclHV0KIPOuiZqrRV0JIp"
access_token = "818333233754451968-F5XS0rD9rAKLKInq9ucoHHBpmnHnUOa"
access_token_secret = "KgWgTUZUZjZY6U1m4gBlr4KsM2jaRVIy549mkkz6Y1j6b"


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

L  = []
author=[]


class CustomStreamListener(tweepy.StreamListener):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.limit = 10
        
    def on_status(self, status):
            #print(status.text)
            global L
            L.append(status.text) 
            author.append(status.user.screen_name)
            self.counter += 1
            if self.counter < self.limit:
                return True
            else:
                return False

    def on_error(self, status_code):
        print >> sys.stderr, 'Encountered error with status code:', status_code
        return True # Don't kill the stream

    def on_timeout(self):
        print >> sys.stderr, 'Timeout...'
        return False # Don't kill the stream

    
    
    
sapi = tweepy.streaming.Stream(auth, CustomStreamListener())    
sapi.filter(locations=[-74.36141,40.55905,-73.704977,41.01758])
main = pd.DataFrame(L,columns=['Tweet'])


#print(main)
#print(author)

print(type(main))



I_tweets=main.values.tolist()
          


# In[19]:



posts=[]
for row in main["Tweet"]:
    #tokenize words
    words = wpt.tokenize(row)
    #remove punctuations
    clean_words = [word.lower() for word in words if word not in set(string.punctuation)]
    #remove stop words
    english_stops = nltk.corpus.stopwords.words('english')
    characters_to_remove = ["''",'``',"rt","https","’","“","”","\u200b","--","n't","'s","...","//t.c" ]
    clean_words = [word for word in clean_words if word not in english_stops]
    clean_words = [word for word in clean_words if word not in set(characters_to_remove)]
    #Lematise words
    lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]
    #print(lemma_list)
    posts.append(lemma_list)
    
    
refine=[]
    
for x in posts:
    s=''
    for y in x:
            s=s+' '+y
    s = re.sub('[^A-Za-z0-9" "]+', '', s)
    s=s.lstrip()
    refine.append(s)    

    
#print(refine)    


# In[20]:



#re_tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
re_tv_matrix = tv.transform(refine)
re_tv_matrix = re_tv_matrix.toarray()

#print(re_tv_matrix)

OUTPUT=svm.predict(re_tv_matrix)


SEVERITY=svm.predict_proba(re_tv_matrix)

print(type(SEVERITY[0][1]))
#print(OUTPUT)
#print(SEVERITY)


# In[21]:


ind=list()
for i in main['Tweet']:
    index = list(main['Tweet']).index(i)
    if(OUTPUT[index]=="Bullying"):
        ind.append(author[index])
        
        
#print(ind)        

limit=0

timeline=[]

CRIM_SCORE=[]
CRIM_NAME=[]

F_SCORE=[]
F_NAME=[]

for i in ind:
    #print(i)
    #print(limit)
    for po in tweepy.Cursor(api.user_timeline, screen_name=i, tweet_mode="extended",exclude='retweets').items():
        if(limit<10):
            if (not po.retweeted) and ('RT @' not in po.full_text):
                timeline.append(po.full_text)
                #print(po.full_text)
                limit=limit+1
        else:
            limit=0
            break
        
    final= pd.DataFrame(timeline,columns=['Tweet'])
    posts_culprit=[]
    for row in final["Tweet"]:
        #tokenize words
        words = wpt.tokenize(row)
        #remove punctuations
        clean_words = [word.lower() for word in words if word not in set(string.punctuation)]
        #remove stop words
        english_stops = nltk.corpus.stopwords.words('english')
        characters_to_remove = ["''",'``',"rt","https","’","“","”","\u200b","--","n't","'s","...","//t.c" ]
        clean_words = [word for word in clean_words if word not in english_stops]
        clean_words = [word for word in clean_words if word not in set(characters_to_remove)]
        #Lematise words
        lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]
        #print(lemma_list)
        posts_culprit.append(lemma_list)


    refine_culprit=[]

    for x in posts:
        s=''
        for y in x:
                s=s+' '+y
        s = re.sub('[^A-Za-z0-9" "]+', '', s)
        s=s.lstrip()
        refine_culprit.append(s)    


    #print(refine_culprit)    

    final_tv_matrix = tv.transform(refine)
    final_tv_matrix = final_tv_matrix.toarray()

    #print(re_tv_matrix)

    OUTPUT_FINAL=svm.predict(final_tv_matrix)


    SEVERITY_FINAL=svm.predict_proba(final_tv_matrix)


    
    #print(OUTPUT_FINAL)
    #print(SEVERITY_FINAL)
    
    for j in range(SEVERITY.shape[0]):
        x
        if(OUTPUT[j]=="Bullying"):    
            x=SEVERITY[j][0]
            #print(x,i)
            for k in range(SEVERITY_FINAL.shape[0]):
                if SEVERITY_FINAL[k][0]>0.5:
                    #print("ADD",SEVERITY_FINAL[k][0])
                    x=x+SEVERITY_FINAL[k][0]
                    #print("ADD")
                else:
                    #print("SUB",SEVERITY_FINAL[k][0])
                    x=x-SEVERITY_FINAL[k][1]
                    #print("SUB")
                #print(x)    
                break
            CRIM_SCORE.append(x)
            CRIM_NAME.append(i)    
    
            
    
    for m in CRIM_SCORE: 
        if m not in F_SCORE: 
            F_SCORE.append(m) 
        
    for m in CRIM_NAME: 
        if m not in F_NAME: 
            F_NAME.append(m) 
        



print((F_SCORE)) 
print(len(F_NAME))





connection = pymysql.connect(host='localhost',
                                    user='root',
                                    password='',
                                    db='cyberproject',
                                    charset='utf8mb4',
                                    cursorclass=pymysql.cursors.DictCursor)

try:
    with connection.cursor() as cursor:
                # Create a new record
      
        #for i in range(len(F_SCORE)):
           # cursor.execute("INSERT INTO tweets(score) values(%s)",(F_SCORE[i]))
            #cursor.execute(sql, (float(F_SCORE[i])))
        for i in range(len(author)):
            cursor.execute("INSERT INTO tweetdb(name,tweets,class) values(%s,%s,%s)",(author[i],I_tweets[i],OUTPUT[i]) )
      
    connection.commit()

    #with connection.cursor() as cursor:
     #           # Read a single record
      #  sql = "SELECT `name` FROM `tweets` WHERE `score`=%f"
       # for i in range(len(F_SCORE)):
        #    cursor.execute(sql, (float(F_SCORE[i])))
            
        #result = cursor.fetchone()
        #print(result)
finally:
            connection.close()
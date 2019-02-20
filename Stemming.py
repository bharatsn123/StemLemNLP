//Source Code for Performing Stemming and Lemmatization on 10 txt files 
import nltk 
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
def det():
   cnt = []
   for i in range(1,11):
        with open (f"a{i}.txt", "r") as myfile:
            data=myfile.read().replace('\n',' ')
            unique_string = data.replace(".","")
            cnt.append(unique_string)
   print(cnt)
   //using Lanstemmer to stem the words
   lanstemmer = LancasterStemmer()
   final = []
   stop_words = set(stopwords.words('english'))
   for w in cnt: 
       s = " "
       terms = w.split(" ")
       inter = []
       for i in terms:
           if i not in stop_words:
                if i.endswith('ing'):
                         i = lanstemmer.stem(i)
                if i not in inter:
                    inter.append(i)
                    s += i + " "
                    
       final.append(s)
   print(final) //printing the result on console
   
   //transfering the resultant matrix to .csv file
   vec = CountVectorizer()
   X = vec.fit_transform(final)
   df = pd.DataFrame(X.toarray(),columns=vec.get_feature_names())
   print("\nDocument-Term Matrix\n")
   print(df)
   print("\nSize = " + str(df.size) +"\n")
   print("\nShape = " + str(df.shape) + "\n")
   df.to_csv('matrix.csv')
det()

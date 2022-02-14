#Spam Message detection

#Importing requrie model
import pickle
import streamlit as st
import nltk
import sklearn
import string
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#Loading the saved model
loaded_model = pickle.load(open("trained_model.sav","rb"))
tfidf = pickle.load(open("Vectorizer.sav", "rb"))

#Preprocessing
# Working with complete text like removing punctuation, stopword
# Creating an object for poter stem which convert word to it root form.
PS = PorterStemmer()

def transform_text(text):
    # Convert complet text into text
    text = text.lower()
  
    # Preprossing the text into word
    text = nltk.word_tokenize(text)

    #Empty list
    mylist = []

    #Collect only alphanumeric word from text
    for i in text: 
        if i.isalnum():
            mylist.append(i)
    
    text = mylist[:]
    print(text)
    mylist.clear()
    
    #Removing all stopword from text
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            mylist.append(i)
  
    mylist.clear()

    #Take only root word 
    for i in text:
        mylist.append(PS.stem(i))
  
    return " ".join(mylist)

def main():

    st.markdown(f'<h1 style="color:#800080;font-size:30px;">{"Email/SMS Spam Classifier"}</h1>', unsafe_allow_html=True)

    #st.markdown(f'<h1 style="color:##808080;font-size:20px;">{"Enter your Message"}</h1>', unsafe_allow_html=True) 

    input_text = st.text_area("Enter your Message")
    
    if st.button("Predict"):

        #1.Preprocessing
        text = transform_text(input_text)

        #2.Vectorizer
        Vectorize_text = tfidf.transform([text])

        #3 Predict
        output = loaded_model.predict(Vectorize_text)[0]
        
        if input_text:
            if output == 1:
                #st.header("Spam")
                st.markdown(f'<h1 style="color:##808080;font-size:20px;">{"Spam"}</h1>', unsafe_allow_html=True) 
            else:
                #st.header("Not Spam")
                st.markdown(f'<h1 style="color:##808080;font-size:20px;">{"Not Spam"}</h1>', unsafe_allow_html=True) 
        else:
            #st.header("Please enter Message")
            st.markdown(f'<h1 style="color:##808080;font-size:20px;">{"Please enter Message"}</h1>', unsafe_allow_html=True) 
            
if __name__ == "__main__":
    main()
    


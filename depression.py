import pickle
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(layout="wide", page_icon=":heart:")

# Add custom styles
st.markdown("""
<style>
    /* Button styles */
    div.stButton > button:first-child {
        background-color: #FF69B4;  /* HotPink color */
        color: white;
        font-size: 18px;
        height: 3em;
        width: 20em;
        border-radius: 15px;
        border: 2px solid white;
        font-weight: bold;
    }

    /* Background image */
    .reportview-container {
        background: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTI8DRxgEP4PMaChfWJQKulfwMWdF486bB0SF0ZHXkgS5z4gc2Jd7EGKC8-gjjKWNxEUlQ&usqp=CAU");
        background-size: cover;
    }

    /* Custom styles for text */
    h1 {
        color: #FFD700;  /* Gold color */
        font-weight: bold;
        text-shadow: 2px 2px black;
    }

    p {
        color: #ADFF2F;  /* GreenYellow color */
        font-size: 18px;
    }

    /* Disclaimer Styles */
    .sidebar .markdown-text-container {
        background-color: #FF4500;  /* Orangered color */
        padding: 10px;
        border-radius: 5px;
    }

    .sidebar .markdown-text-container span {
        color: yellow;
        font-weight: bold;
    }

</style>
""",
            unsafe_allow_html=True
)

st.image(r"https://www.spbh.org/wp-content/uploads/2020/09/stop-suicide.jpg", width=500, use_column_width=True)

# Enhanced Disclaimer Section
st.sidebar.markdown('### **Disclaimer**')
st.sidebar.markdown("""
<span>
This is a research work by OdumareTech.
It's important to understand that this tool can generate inaccurate predictions. 
Always consult with a professional for serious concerns. 
Note: This application does not save any data inputted in it.
</span>
""", unsafe_allow_html=True)

# Load the saved model
with open('logistic_re.pkl', 'rb') as file:
    classifier, vectorizer = pickle.load(file)

# Define a Streamlit app
def app():
    st.title("OdumareTech Suicide Risk Predictor")
    
    # Larger text input box
    user_input = st.text_area("Please enter the text:", height=200)
    
    if st.button('Predict'):
        # Vectorize the input text
        text_vectorized = vectorizer.transform([user_input])

        # Make the prediction using the trained model
        prediction = classifier.predict(text_vectorized)[0]

        # Set the color of the prediction text
        color = "#FF4500" if prediction == 'suicide' else "#3CB371"  # Use Orangered for suicide and MediumSeaGreen for non-suicide

        # Display the prediction result with the specified color
        if color == "#FF4500":
            st.write('The model predicts that the writer of this text is likely depressed and has a high risk of suicide.')
        else:
            st.write('The model predicts that the writer of this text is not depressed and has no intention of suicide.')

        # Display the prediction result with the specified color
        st.markdown(f'<p style="color:{color}; font-size: 24px; font-weight: bold;">{prediction}</p>', unsafe_allow_html=True)

if __name__ == '__main__':
    app()

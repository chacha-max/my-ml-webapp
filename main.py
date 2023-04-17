import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image

# loading in the model to predict on the data
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)


def welcome():
    return 'welcome all'


# defining the function which will make the prediction using
# the data which the user inputs
def prediction(gre, gpa, rank):
    rank_1 = rank_2 = rank_3 = rank_4 = 0
    if rank == 1:
        rank_1 = 1
    elif rank == 2:
        rank_2 = 1
    elif rank == 3:
        rank_3 = 1
    elif rank == 4:
        rank_4 = 1

    prediction = classifier.predict(
        [[gre, gpa, rank_1, rank_2, rank_3, rank_4]])
    print(prediction)
    return prediction


# this is the main function in which we define our webpage
def main():
    # giving the webpage a title
    st.title("Student Admission Prediction")

    st.markdown(
        f"""
             <style>
             .stApp {{
                 background: url("https://heiapply.com/wp-content/uploads/2017/06/online-admissions-universities_video_background-1024x467.jpg");
                 background-size: cover
             }}
             </style>
             """,
        unsafe_allow_html=True
    )

    # Set the button color to gold
    st.markdown(
        """
        <style>
        .stButton button {
            background-color: gold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # here we define some of the front end elements of the web page like
    # the font and background color, the padding and the text to be displayed
    html_temp = """
	<div style ="background-color:yellow;padding:13px">
	<h1 style ="color:black;text-align:center;">Streamlit Student Admission Classifier ML App </h1>
	</div>
	"""

    # this line allows us to display the front end aspects we have
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html=True)

    # the following lines create text boxes in which the user can enter
    # the data required to make the prediction
    gre = st.number_input("**GRE**:", min_value=200, max_value=990, step=1, help="Enter a value between 200 and 990")

    gpa = st.slider('**GPA**:', min_value=0.0, max_value=4.0, value=2.0, step=0.01)

    rank = st.radio('Choose **Institution Rank**:', options=('1', '2', '3', '4'))
    st.write('Please note that institutions with a rank of 1 have the highest prestige, while those with a rank of 4 have the lowest.')

    result = "The output will be displayed here."

    # Create the prediction button with a bright yellow color
    prediction_button = st.button("Predict", key="predict", help="Click this button to make a prediction",
                                  on_click=None, args=None, kwargs=None,
                                  )

    # the below line ensures that when the button called 'Predict' is clicked,
    # the prediction function defined above is called to make the prediction
    # and store it in the variable result
    # Change the background color of the success message to green

    if prediction_button:
        result = prediction(gre, gpa, rank)
        if result:
            result = "Congratulations! You are admitted to graduate school."
        else:
            result = "Unfortunately, you are not admitted to graduate school."
    st.success(result)

    st.markdown(
        """<style>
            .css-2trqyj {
                background-color: "#0083B8";
            }
        </style>""",
        unsafe_allow_html=True,
    )


if __name__ == '__main__':
    main()
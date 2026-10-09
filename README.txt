WHO'S A LINKEDIN USER? - STREAMLIT APP
Programming II final project, Group 3: Zara, Ilknur, Sean, Finley, and Andre

FILES
app.py                        the Streamlit app (builds the same model as our notebook)
social_media_usage.csv        the Pew survey data from the course
social_media_usage_README.txt the codebook
requirements.txt              the packages the app needs

RUN THE APP
Open a terminal in this folder and run:

    pip install -r requirements.txt
    streamlit run app.py

The app opens in your browser at http://localhost:8501

USING THE APP
Pick a person's details in the sidebar. The prediction (LinkedIn user or not, and the
probability) updates as soon as an input changes. The page also shows how age changes
the prediction, the assignment's age 42 vs 82 example, LinkedIn use by income and
education in the survey, and the model's test set results.

The data come from Pew and are for educational use only.

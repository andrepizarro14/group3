WHO'S A LINKEDIN USER? - STREAMLIT APP
Programming II final project, Group 3: Zara, Ilknur, Sean, Finley, and Andre

FILES
app.py            the Streamlit app (builds the same model as our notebook)
data/             social_media_usage.csv and its codebook, as supplied for the course
requirements.txt  the Python packages the app needs
run_local.sh      starts the app on Finley's Mac

RUN THE APP
On Finley's Mac, open Terminal in this folder and run:   ./run_local.sh
Then open http://127.0.0.1:8502 in a browser.

On any other computer, open a terminal in this folder and run:
pip install -r requirements.txt
streamlit run app.py

USING THE APP
Choose a person's details in the sidebar. The prediction (LinkedIn user or not, and
the probability) updates as soon as an input changes. The page also shows how age
changes the prediction, the assignment's age 42 vs age 82 example, LinkedIn use by
income and education in the survey, and the model's test-set results.

The data come from Pew and are for educational use only.

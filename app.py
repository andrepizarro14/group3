### Who's a LinkedIn user? Streamlit app for the Programming II final project (Group 3)

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


######### 1 Build the model (same steps as our notebook)

def clean_sm(x):
    # if the value is 1 keep it as 1, anything else becomes 0
    x = np.where(x == 1, 1, 0)
    return x

s = pd.read_csv("data/social_media_usage.csv")

ss = pd.DataFrame({
    "sm_li": clean_sm(s["web1h"]),
    "income": np.where(s["income"] > 9, np.nan, s["income"]),
    "education": np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent": np.where(s["par"] == 1, 1, 0),
    "married": np.where(s["marital"] == 1, 1, 0),
    "female": np.where(s["gender"] == 2, 1, 0),
    "age": np.where(s["age"] > 98, np.nan, s["age"])
})

ss = ss.dropna()
ss = ss.astype("int")

y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

X_train, X_test, y_train, y_test = train_test_split(X.values,
                                                    y,
                                                    stratify=y,
                                                    test_size=0.2,
                                                    random_state=987)

lr = LogisticRegression(class_weight="balanced")
lr.fit(X_train, y_train)


######### 2 Title

st.markdown("# Who's a LinkedIn user?")
st.markdown("#### Group 3: Zara, Ilknur, Sean, Finley, and Andre")
st.write("Enter a person's details in the sidebar and the app predicts whether they use LinkedIn. "
         "The prediction updates as soon as you change an input. The model is the logistic regression "
         "from our notebook, trained on the 2021 Pew social media survey.")


######### 3 User inputs in the sidebar

income_options = {
    "Less than 10k": 1,
    "10k to under 20k": 2,
    "20k to under 30k": 3,
    "30k to under 40k": 4,
    "40k to under 50k": 5,
    "50k to under 75k": 6,
    "75k to under 100k": 7,
    "100k to under 150k": 8,
    "150k or more": 9
}

education_options = {
    "Less than high school": 1,
    "High school incomplete": 2,
    "High school graduate": 3,
    "Some college, no degree": 4,
    "Associate degree": 5,
    "Bachelor's degree": 6,
    "Some postgrad, no degree": 7,
    "Postgrad or professional degree": 8
}

with st.sidebar:
    st.markdown("## Person's details")
    income_label = st.selectbox("Household income", options=list(income_options.keys()), index=7)
    education_label = st.selectbox("Education", options=list(education_options.keys()), index=6)
    parent_label = st.selectbox("Parent of a child under 18 living at home?", options=["No", "Yes"])
    married_label = st.selectbox("Married?", options=["No", "Yes"], index=1)
    gender_label = st.selectbox("Gender", options=["Female", "Male", "Other"])
    age = st.number_input("Age (18 to 97)", 18, 97, 42)

# Convert the labels into the numbers the model uses
income = income_options[income_label]
education = education_options[education_label]

if parent_label == "Yes":
    parent = 1
else:
    parent = 0

if married_label == "Yes":
    married = 1
else:
    married = 0

if gender_label == "Female":
    female = 1
else:
    female = 0


######### 4 Prediction

# Features in the same order as the model: income, education, parent, married, female, age
person = [income, education, parent, married, female, age]

predicted_class = lr.predict([person])
probs = lr.predict_proba([person])
probability = probs[0][1]

st.markdown("***")
st.markdown("## Prediction")

if predicted_class[0] == 1:
    st.markdown("### Classified as: a LinkedIn user")
else:
    st.markdown("### Classified as: not a LinkedIn user")

st.markdown(f"### Probability of using LinkedIn: {round(probability * 100, 1)}%")
st.write("The model classifies someone as a LinkedIn user when the probability is above 50%.")
st.write(f"Inputs used: income {income}, education {education}, parent {parent}, "
         f"married {married}, female {female}, age {age}")


######### 5 Same person at every age

st.markdown("***")
st.markdown("## How does age change the prediction for this person?")

ages = list(range(18, 98))
age_probs = [lr.predict_proba([[income, education, parent, married, female, each]])[0][1] for each in ages]

by_age = pd.DataFrame({
    "Age": ages,
    "Probability of using LinkedIn": age_probs
})

st.altair_chart(alt.Chart(by_age).mark_circle().encode(
    x="Age",
    y="Probability of using LinkedIn",
    tooltip=["Age", "Probability of using LinkedIn"]).
    properties(title="Same inputs as the sidebar, only age changes"))

st.write("Everything except age stays the same as the sidebar. The probability falls steadily as age goes up.")


######### 6 The two people from the assignment

st.markdown("***")
st.markdown("## The assignment's example: age 42 vs age 82")
st.write("High income (8), high education (7), not a parent, married, female. Only age changes.")

person_42 = [8, 7, 0, 1, 1, 42]
person_82 = [8, 7, 0, 1, 1, 82]
prob_42 = lr.predict_proba([person_42])[0][1]
prob_82 = lr.predict_proba([person_82])[0][1]

example = pd.DataFrame({
    "Age": [42, 82],
    "Predicted class (1 = LinkedIn user)": [lr.predict([person_42])[0], lr.predict([person_82])[0]],
    "Probability of using LinkedIn (%)": [round(prob_42 * 100, 1), round(prob_82 * 100, 1)]
})

st.dataframe(example)
st.write(f"Changing only the age from 42 to 82 lowers the probability by "
         f"{round((prob_42 - prob_82) * 100, 1)} percentage points.")


######### 7 Who uses LinkedIn in the survey

st.markdown("***")
st.markdown("## Who uses LinkedIn in the survey?")
st.write("Share of survey respondents who use LinkedIn. Income and education are the features most "
         "strongly related to LinkedIn use, so they are the most useful for picking target segments.")

li_by_income = ss.groupby("income", as_index=False)[["sm_li"]].mean()
income_chart_data = pd.DataFrame({
    "Income bracket (1 = under 10k, 9 = 150k+)": li_by_income["income"],
    "Share using LinkedIn": li_by_income["sm_li"]
})

st.altair_chart(alt.Chart(income_chart_data).mark_bar().encode(
    x="Income bracket (1 = under 10k, 9 = 150k+):N",
    y="Share using LinkedIn").
    properties(title="LinkedIn use jumps once household income passes 75k (bracket 7)"))

li_by_educ = ss.groupby("education", as_index=False)[["sm_li"]].mean()
educ_chart_data = pd.DataFrame({
    "Education (1 = less than high school, 8 = grad degree)": li_by_educ["education"],
    "Share using LinkedIn": li_by_educ["sm_li"]
})

st.altair_chart(alt.Chart(educ_chart_data).mark_bar().encode(
    x="Education (1 = less than high school, 8 = grad degree):N",
    y="Share using LinkedIn").
    properties(title="LinkedIn use is highest from a bachelor's degree (6) upward"))


######### 8 How good is the model?

st.markdown("***")
st.markdown("## How good is the model?")
st.write("Results on the 252 survey respondents held out for testing (20% of the data).")

y_pred = lr.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

tn = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tp = cm[1][1]

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 score"],
    "Value": [round(accuracy, 3), round(precision, 3), round(recall, 3), round(f1, 3)]
})

st.dataframe(metrics)

if st.checkbox("Show the confusion matrix"):
    cm_df = pd.DataFrame(cm,
                         columns=["Predicted: not a user", "Predicted: LinkedIn user"],
                         index=["Actual: not a user", "Actual: LinkedIn user"])
    st.dataframe(cm_df.style.background_gradient(cmap="Blues"))
    st.write(f"The model finds {tp} of the {tp + fn} real LinkedIn users (recall). "
             f"{fp} people are predicted to be users but are not (false positives), "
             f"and {fn} real users are missed (false negatives).")


######### 9 Notes

st.markdown("***")
st.markdown("## Notes")
st.markdown("""
+ The data is a Pew survey from early 2021, so it describes those respondents, not who uses LinkedIn today.
+ Anyone who answered "don't know" or "refused" is coded as 0 (not a user) for the target, and the same goes for parent, married and female.
+ Income and education are survey categories, not dollar amounts or years of school.
+ The data is for educational use only.
""")

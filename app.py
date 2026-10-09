### Who's a LinkedIn user? Streamlit app (Programming II final project, Group 3)

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


######### 1 Build the model (same steps as the notebook)

def clean_sm(x):
    # if the value is 1 keep it as 1, anything else becomes 0
    x = np.where(x == 1, 1, 0)
    return x

s = pd.read_csv("social_media_usage.csv")

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

# class_weight was not covered in class, we looked it up in the sklearn docs
lr = LogisticRegression(class_weight="balanced")
lr.fit(X_train, y_train)


######### 2 Title

st.markdown("# Who's a LinkedIn user?")
st.markdown("#### Group 3: Zara, Ilknur, Sean, Finley, and Andre")
st.write("Pick a person's details in the sidebar and the app predicts whether they use LinkedIn. "
         "The model is the logistic regression from our notebook, trained on the 2021 Pew survey.")


######### 3 User inputs in the sidebar

with st.sidebar:
    st.markdown("## Person's details")

    inc = st.selectbox("Household income",
                       options=["Less than 10k",
                                "10k to under 20k",
                                "20k to under 30k",
                                "30k to under 40k",
                                "40k to under 50k",
                                "50k to under 75k",
                                "75k to under 100k",
                                "100k to under 150k",
                                "150k or more"])

    educ = st.selectbox("Education",
                        options=["Less than high school",
                                 "High school incomplete",
                                 "High school graduate",
                                 "Some college, no degree",
                                 "Associate degree",
                                 "Bachelor's degree",
                                 "Some postgrad, no degree",
                                 "Postgrad or professional degree"])

    par = st.selectbox("Parent of a child under 18 at home?", options=["No", "Yes"])
    mar = st.selectbox("Married?", options=["No", "Yes"])
    gen = st.selectbox("Gender", options=["Male", "Female", "Other"])
    age = st.number_input("Age (18 to 97)", 18, 97)

# Turn the labels into the numbers the model was trained on

# Income (1 to 9)
if inc == "Less than 10k":
    income = 1
elif inc == "10k to under 20k":
    income = 2
elif inc == "20k to under 30k":
    income = 3
elif inc == "30k to under 40k":
    income = 4
elif inc == "40k to under 50k":
    income = 5
elif inc == "50k to under 75k":
    income = 6
elif inc == "75k to under 100k":
    income = 7
elif inc == "100k to under 150k":
    income = 8
else:
    income = 9

# Education (1 to 8)
if educ == "Less than high school":
    education = 1
elif educ == "High school incomplete":
    education = 2
elif educ == "High school graduate":
    education = 3
elif educ == "Some college, no degree":
    education = 4
elif educ == "Associate degree":
    education = 5
elif educ == "Bachelor's degree":
    education = 6
elif educ == "Some postgrad, no degree":
    education = 7
else:
    education = 8

# Parent
if par == "Yes":
    parent = 1
else:
    parent = 0

# Married
if mar == "Yes":
    married = 1
else:
    married = 0

# Female
if gen == "Female":
    female = 1
else:
    female = 0


######### 4 Prediction

# Same feature order as the model: income, education, parent, married, female, age
person = [income, education, parent, married, female, age]

predicted_class = lr.predict([person])
probs = lr.predict_proba([person])

st.markdown("***")
st.markdown("## Prediction")

if predicted_class[0] == 1:
    st.markdown("### Classified as: a LinkedIn user")
else:
    st.markdown("### Classified as: not a LinkedIn user")

st.markdown(f"### Probability of using LinkedIn: {round(probs[0][1] * 100, 1)}%")
st.write("Anyone with a probability above 50% is classified as a LinkedIn user.")
st.write(f"This person is coded as: income {income}, education {education}, parent {parent}, "
         f"married {married}, female {female}, age {age}")


######### 5 Same person at every age

st.markdown("***")
st.markdown("## How does age change the prediction for this person?")

ages = list(range(18, 98))
age_probs = [lr.predict_proba([[income, education, parent, married, female, each]])[0][1] for each in ages]

by_age = pd.DataFrame({
    "age": ages,
    "prob_linkedin": age_probs
})

age_plot = alt.Chart(by_age).mark_circle().encode(
    x="age",
    y="prob_linkedin",
    tooltip=["age", "prob_linkedin"]).\
properties(title="Same details as the sidebar, only age changes")

st.altair_chart(age_plot)
st.write("The probability falls as age goes up. Everything except age stays the same as the sidebar.")


######### 6 The two people from the assignment

st.markdown("***")
st.markdown("## The assignment's example: age 42 vs age 82")
st.write("Income 8, education 7, not a parent, married, female. Only age changes.")

person_42 = [8, 7, 0, 1, 1, 42]
person_82 = [8, 7, 0, 1, 1, 82]

prob_42 = lr.predict_proba([person_42])[0][1]
prob_82 = lr.predict_proba([person_82])[0][1]

example = pd.DataFrame({
    "age": [42, 82],
    "predicted_class": [lr.predict([person_42])[0], lr.predict([person_82])[0]],
    "prob_linkedin": [round(prob_42, 3), round(prob_82, 3)]
})

st.dataframe(example)
st.write(f"Changing only the age from 42 to 82 lowers the probability by "
         f"{round((prob_42 - prob_82) * 100, 1)} percentage points.")


######### 7 Who uses LinkedIn in the survey

st.markdown("***")
st.markdown("## Who uses LinkedIn in the survey?")
st.write("Share of respondents who use LinkedIn by income bracket (1 = under 10k, 9 = 150k or more) "
         "and by education level (1 = less than high school, 8 = graduate degree).")

li_by_income = ss.groupby("income", as_index=False)[["sm_li"]].mean()

income_plot = alt.Chart(li_by_income).mark_bar().encode(
    x="income:N",
    y="sm_li").\
properties(title="LinkedIn use jumps once household income passes 75k (bracket 7)")

st.altair_chart(income_plot)

li_by_educ = ss.groupby("education", as_index=False)[["sm_li"]].mean()

educ_plot = alt.Chart(li_by_educ).mark_bar().encode(
    x="education:N",
    y="sm_li").\
properties(title="LinkedIn use is highest from a bachelor's degree (6) upward")

st.altair_chart(educ_plot)


######### 8 How good is the model?

st.markdown("***")
st.markdown("## How good is the model?")
st.write("Results on the 252 respondents held out for testing (20% of the data).")

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
    "metric": ["Accuracy", "Precision", "Recall", "F1 score"],
    "value": [round(accuracy, 3), round(precision, 3), round(recall, 3), round(f1, 3)]
})

st.dataframe(metrics)

if st.checkbox("Show the confusion matrix"):
    cm_df = pd.DataFrame(cm,
                         columns=["Predicted: not a user", "Predicted: LinkedIn user"],
                         index=["Actual: not a user", "Actual: LinkedIn user"])
    st.dataframe(cm_df)
    st.write(f"The model finds {tp} of the {tp + fn} real LinkedIn users. "
             f"{fp} people are predicted to be users but are not, and {fn} real users are missed.")


######### 9 Notes

st.markdown("***")
st.markdown("## Notes")
st.markdown("""
+ The data is a Pew survey from early 2021, so it describes those respondents, not who uses LinkedIn today.
+ Anyone who answered "don't know" or "refused" is coded as 0 for the target, and the same goes for parent, married and female.
+ Income and education are survey categories, not dollar amounts or years of school.
+ The data is for educational use only.
""")

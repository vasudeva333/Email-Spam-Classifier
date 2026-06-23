# Email-Spam-Classifier

INTERN ID: CITS5352
NAME: VASUDEVA REDDY
NO.OF WEEKS: 4

PROJECT NAME :Email-Spam-Classifier 

PROJECT SCOPE:

   Main Goal
       The goal of this project is to create an smart tool that reads text messages. It automatically labels them as Spam (junk/scams) or Ham (safe/good emails). This keeps a user's inbox clean and safe from bad links.
  What the Project Does and Does Not DoWhat is Included:
  Cleaning Data:Fixing missing blanks in the email dataset.
  Turning Words into Numbers: Using a tool called TF-IDF to turn text into math that computers understand.
  Smart Training: Teaching a Logistic Regression model to find clues that separate good mail from bad mail.
  Testing: Checking accuracy scores to see how well the system works on new data.
  Live Prediction: Letting a user paste a new sentence to get an instant "Spam" or "Ham" answer.
  What is Excluded:
  It does not connect to real apps like Gmail or Outlook.It does not have visual buttons, screens, or a mobile app layout.It cannot scan picture attachments or files.It cannot update itself automatically with new data from the internet.


source code:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load and clean data
df = pd.read_csv('mail_data.csv').fillna('')
X = df['Message']
Y = df['Category'].map({'spam': 0, 'ham': 1})

# Split into Train and Test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Convert text to numbers
vec = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_vec, Y_train)

# Print final test accuracy
print("Test Accuracy:", model.score(X_test_vec, Y_test))

# Test with custom email
new_mail = ["Free entry in 2 a wkly comp to win FA Cup final..."]
pred = model.predict(vec.transform(new_mail))
print("Prediction:", "Ham mail" if pred[0] == 1 else "Spam mail")





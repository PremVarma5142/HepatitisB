import hashlib
import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.figure(figsize=(6,4))
st.set_option('deprecation.showPyplotGlobalUse', False)

df = pd.read_csv("hepatitis.csv")
from managed_DB import *

def generate_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def verify_hashes(password ,hashed_text):
    if generate_hashes(password) == hashed_text:
        return hashed_text
    return False


def main():

    st.title("Disease Mortality Prediction App by Prem Varma")
    menu = ['Home','Login','SignUp']
    submenu = ['Plot','Prediction']
    choice = st.sidebar.selectbox('Menu',menu)
    if choice == 'Home':
        st.subheader('Home')
        st.text("What is HepatitisB? ")
        st.write("""
        A serious liver infection caused by the hepatitis B virus that's easily preventable by a vaccine.
This disease is most commonly spread by exposure to infected bodily fluids.
Symptoms are variable and include yellowing of the eyes, abdominal pain and dark urine. Some people, particularly children, don't experience any symptoms. In chronic cases, liver failure, cancer or scarring can occur.
The condition often clears up on its own. Chronic cases require medication and possibly a liver transplant.

        """)

    elif choice == 'Login':
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password",type='password')

        if st.sidebar.checkbox('Login'):
            create_usertable()
            hashed_pass = generate_hashes(password)
            result = login_user(username,verify_hashes(password,hashed_pass))
            # if password == 'Prem@123':
            if result:
                st.success("Welcome {}".format(username))
                activity = st.selectbox('Activity',submenu)
                if activity == 'Plot':
                    st.subheader("Data is Plot")

                    df.dropna(inplace=True)


                    st.dataframe(df)



                    df['class'].value_counts().plot(kind='bar')
                    st.pyplot()
                    if st.checkbox("Area Chart"):
                        clear_columns = df.columns.to_list()
                        feat_choices = st.multiselect("Choose Features",clear_columns)
                        new_dff = df[feat_choices]
                        st.area_chart(new_dff)
                elif activity == 'Prediction':
                    st.subheader("Predictive Analysis")



                    x = df[['age', 'sex', 'steroid', 'antivirals',
                                                'spiders', 'ascites', 'varices',
                                                'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime', 'histology']].values
                    y = df[['class']].values

                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

                    st.text("SEX == Male : 1 , Female : 2")
                    st.text("Yes : 2 , No : 1")
                    def get_user_input():
                        age = st.number_input('Age',7,80)
                        sex = st.number_input('Sex',1,2)
                        steroid = st.number_input('steroid',1,2)
                        antivirals = st.number_input('antivirals',1,2)
                        spiders = st.number_input('Is presence of spider?',1,2)
                        ascites = st.number_input('Asciets',1,2)
                        varices = st.number_input('Presence of varices',1,2)
                        bilirubin = st.sidebar.slider('Bilirubin content',0.0,4.0,8.0)
                        alk_phosphate =st.sidebar.slider('Alk phosphate',0.0,196.0,296.0)
                        sgot = st.sidebar.slider('sgot',0.0,300.0,648.0)
                        albumin =st.sidebar.slider('albumin',0.0,3.4,6.4)
                        protime = st.sidebar.slider('protime',0.0,50.0,100.0)
                        histology =st.number_input('Histology',1,2)

                        user_data = {'age':age,
                                         'sex':sex,
                                         'steroid':steroid,
                                         'antivirals':antivirals,
                                         'spiders':spiders,
                                         'ascites':ascites,
                                         'varices':varices,
                                         'bilirubin':bilirubin,
                                         'alk_phosphate':alk_phosphate,
                                         'sgot':sgot,
                                         'albumin':albumin,
                                         'protime':protime,
                                         'histology':histology}
                        features = pd.DataFrame(user_data, index=[0])
                        return features
                    user_input = get_user_input()

                    #
                    # pretty_result = {'age':age, 'sex':sex, 'steroid':steroid, 'antivirals':antivirals,'spiders':spiders,'ascites':ascites,'varices':varices,'bilirubin':bilirubin, 'alk_phosphate':alk_phosphate, 'sgot':sgot, 'albumin':albumin, 'protime':protime, 'histology':histology}
                    # st.json(pretty_result)
                    # simple_sample = np.array(user_input).reshape(1,-1)
                    st.subheader('User Input')
                    st.write(user_input)

                    model_choice = st.selectbox("Select Model",["KNN","RFC"])
                    if st.button("Predict"):
                        if model_choice == "KNN":

                            #
                            # KN = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
                            # KN.fit(x_train, y_train.ravel())
                            from sklearn.neighbors import KNeighborsClassifier
                            KNeighborsClassifier = KNeighborsClassifier()
                            KNeighborsClassifier.fit(x_train, y_train)
                            # accuracy_score = st.subheader('Model Test Accuracy Score:')
                            # st.write(str(accuracy_score(y_test, KNeighborsClassifier.predict(x_test)) * 100) + '%')
                            from sklearn.metrics import confusion_matrix
                            # cm = confusion_matrix()

                            cm = confusion_matrix(y_test,KNeighborsClassifier.predict(x_test))
                            tn, fp, fn, tp = confusion_matrix(y_test,KNeighborsClassifier.predict(x_test)).ravel()
                            test_score = (tp + tn) / (tp + tn + fp + fn)
                            st.write(cm)
                            st.write("Model Accuracy is: {}".format(test_score))

                            from sklearn.metrics import classification_report
                            from sklearn.metrics import accuracy_score

                            st.write(classification_report(y_test,KNeighborsClassifier.predict(x_test)))
                            st.write(accuracy_score(y_test,KNeighborsClassifier.predict(x_test)))

                            prediction = KNeighborsClassifier.predict(user_input)

                            if prediction == 1:
                                st.write("You Don't have HepatitisB")
                            else:
                                st.write("Sorry you have HepatitisB")

                        if model_choice == "RFC":


                            from sklearn.ensemble import RandomForestClassifier

                            RandomForestClassifier = RandomForestClassifier()
                            RandomForestClassifier.fit(x_train, y_train)

                            # show the model
                            # st.subheader('Model Test Accuracy Score:')
                            # st.write(str(accuracy_score(y_test, RandomForestClassifier.predict(x_test)) * 100) + '%')

                            # store the model prediction
                            from sklearn.metrics import confusion_matrix
                            # cm = confusion_matrix()

                            cm = confusion_matrix(y_test, RandomForestClassifier.predict(x_test))
                            tn, fp, fn, tp = confusion_matrix(y_test,RandomForestClassifier.predict(x_test)).ravel()
                            test_score = (tp + tn) / (tp + tn + fp + fn)
                            st.write(cm)
                            st.write("Model Accuracy is: {}".format(test_score))

                            from sklearn.metrics import classification_report
                            from sklearn.metrics import accuracy_score

                            st.write(classification_report(y_test, RandomForestClassifier.predict(x_test)))
                            st.write(accuracy_score(y_test,RandomForestClassifier.predict(x_test)))



                            prediction = RandomForestClassifier.predict(user_input)

                            if prediction == 1:
                                st.write("You Don't have HepatitisB")
                            else:
                                st.write("Sorry you have HepatitisB")












            else:
                st.warning("Incorrect Username/Password")

    elif choice == "SignUp":
        new_username = st.text_input("username")
        new_password = st.text_input("Password",type='password')
        confirm_password = st.text_input("Confirm Password",type='password')
        if new_password == confirm_password:
            st.success("Password Confirmed")
        else:
            st.warning("Passwords not the same")
        if st.button("Submit"):
            create_usertable()
            hashed_new_password = generate_hashes(new_password)
            add_userdata(new_username,hashed_new_password)
            st.success("You have successfully created new Account")
            st.info("Login to get started")

if __name__ =='__main__':
    main()

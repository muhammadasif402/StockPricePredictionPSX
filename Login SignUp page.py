import streamlit as st

# Define a dictionary to store user information
user_data = {}

def login():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in user_data and user_data[username]["password"] == password:
            st.success(f"Logged in as: {username}")
            return True
        else:
            st.error("Invalid username or password")
            return False

    return False

def signup():
    st.title("Signup Page")
    new_first_name = st.text_input("**First Name**")
    new_last_name = st.text_input("**Last Name**")
    new_country = st.text_input("**Country Name**")
    new_contact_number = st.text_input("Contact Number (11 digits)")
    new_username = st.text_input("**New Username** ")
    new_password = st.text_input("**New Password** (4-12 characters)", type="password")

    if st.button("Signup"):
        valid_input = True
        
        if not (4 <= len(new_password) <= 12):
            st.error("Password must be 4-12 characters")
            valid_input = False

        if not (len(new_username) <= 30):
            st.error("Username cannot exceed 30 characters")
            valid_input = False

        if not (len(new_first_name) <= 30):
            st.error("First Name cannot exceed 30 characters")
            valid_input = False

        if not (len(new_last_name) <= 30):
            st.error("Last Name cannot exceed 30 characters")
            valid_input = False

        if not (new_country.isalpha() and len(new_country) <= 20):
            st.error("Country must be alphabets only and not exceed 20 characters")
            valid_input = False

        if not (new_contact_number.isdigit() and len(new_contact_number) == 11):
            st.error("Contact Number must be 11 digits")
            valid_input = False

        if valid_input:
            user_data[new_username] = {
                "password": new_password,
                "first_name": new_first_name,
                "last_name": new_last_name,
                "country": new_country,
                "contact_number": new_contact_number
            }
            st.success(f"Signup successful, please login with {new_username}")

def main():
    
    st.title("Authentication For Stock Price Prediction App")
    ans = st.radio("Are You A New User?", ('Yes', 'No'))

    if ans == 'No':
        login()
    elif ans == 'Yes':
        signup()

if __name__ == "__main__":
    main()

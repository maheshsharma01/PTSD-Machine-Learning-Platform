import streamlit_authenticator as stauth

password = "1234"

hashed_password = stauth.Hasher().hash(password)

print(hashed_password)
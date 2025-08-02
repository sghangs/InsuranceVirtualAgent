import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import requests
from src.constant import SIGNUP_ENDPOINT, TOKEN_ENDPOINT, ME_ENDPOINT

st.set_page_config(page_title="Insurance Virtual Agent — Login/Signup", layout="centered", page_icon="🛡️")

class AuthPage:
    def __init__(self):
        self.init_session_state()

    def init_session_state(self):
        if "auth_mode" not in st.session_state:
            st.session_state.auth_mode = "login"
        if "logged_in" not in st.session_state:
            st.session_state.logged_in = False
        if "jwt_token" not in st.session_state:
            st.session_state.jwt_token = None
        if "user_info" not in st.session_state:
            st.session_state.user_info = None

    def render(self):
        self.sidebar_nav()
        if st.session_state.auth_mode == "login":
            self.render_login()
            
        else:
            self.render_signup()
        

    def sidebar_nav(self):
        st.sidebar.title("Navigation")
        if st.session_state.auth_mode == "login":
            st.sidebar.write("Don't have an account?")
            if st.sidebar.button("Go to Sign Up"):
                st.session_state.auth_mode = "signup"
        else:
            st.sidebar.write("Already have an account?")
            if st.sidebar.button("Go to Login"):
                st.session_state.auth_mode = "login"

    def render_signup(self):
        st.title("📝 Sign Up")
        with st.form("signup_form"):
            name = st.text_input("Full Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign Up")
        if submit:
            if not name or not email or not password:
                st.warning("Please fill in all fields.")
                return
            try:
                res = requests.post(SIGNUP_ENDPOINT, json={"name": name, "email": email, "password": password}, timeout=30)
                if res.status_code == 200:
                    st.success("Signup successful! Please log in.")
                    st.session_state.auth_mode = "login"
                    st.stop()
                else:
                    st.error(res.json().get("detail", "Signup failed."))
            except Exception:
                st.error("Could not connect to authentication service.")

    def render_login(self):
        st.title("🔒 Login")
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
        if submit:
            if not email or not password:
                st.warning("Please enter your email and password.")
                return
            try:
                res = requests.post(
                    TOKEN_ENDPOINT,
                    data={"username": email, "password": password},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=30
                )
                if res.status_code == 200:
                    token = res.json()["access_token"]
                    st.session_state.jwt_token = token
                    me_res = requests.get(ME_ENDPOINT, headers={"Authorization": f"Bearer {token}"}, timeout=30)
                    if me_res.status_code == 200:
                        st.session_state.user_info = me_res.json()
                        st.success("Login successful! Redirecting...")
                        import time
                        time.sleep(0.5)  # Optional: short pause for user feedback
                        st.switch_page("pages/insurance_agent.py")  # <--- THIS LINE DOES THE REDIRECT
                    else:
                        st.error("Could not fetch user info.")
                else:
                    st.error(res.json().get("detail", "Login failed."))
            except Exception:
                st.error("Could not connect to authentication service.")

if __name__ == "__main__":
    AuthPage().render()
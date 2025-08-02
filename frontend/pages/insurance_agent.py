import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import re
import uuid
import requests
from src.constant import RAG_ENDPOINTS

st.set_page_config(page_title="Insurance Virtual Agent", layout="centered", page_icon="🛡️")

class InsuranceVirtualAgentPage:
    def __init__(self):
        self.ensure_auth()
        self.init_session_state()

    def ensure_auth(self):
        if not st.session_state.get("jwt_token") or not st.session_state.get("user_info"):
            st.warning("You must log in first via the main page.")
            st.stop()

    def init_session_state(self):
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hi, I'm your insurance policy agent. How may I help you?"
                }
            ]

    def render(self):
        st.title("🏦 Insurance Virtual Agent")
        user_info = st.session_state.get("user_info")
        name = user_info.get("name", "User") if user_info else "User"
        st.markdown(f"Welcome {name}! Please enter your policy number to continue.")

        policy_number = st.text_input("🔢 Enter your Policy Number", placeholder="e.g., AU1234", max_chars=6)
        if policy_number:
            policy_number = policy_number.strip().upper()
            if self.validate_policy(policy_number):
                st.success("Policy number validated.")
                self.chat_interface(policy_number)
            else:
                st.error("❌ Invalid policy number. Please enter a valid policy number (e.g., AU1234).")

    def validate_policy(self, policy_number):
        return bool(re.fullmatch(r"AU\d{4}", policy_number))

    def chat_interface(self, policy_number):
        st.subheader("💬 Ask a question about your policy")
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        prompt = st.chat_input(placeholder="Type your query here...")
        if prompt:
            prompt = prompt.strip()
            if not prompt:
                st.warning("Query cannot be empty.")
            elif len(prompt) > 500:
                st.warning("Query too long. Please keep it under 500 characters.")
            else:
                with st.chat_message("user"):
                    st.markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})

                with st.spinner("Thinking..."):
                    headers = {"Authorization": f"Bearer {st.session_state.jwt_token}"}
                    try:
                        res = requests.post(
                            RAG_ENDPOINTS,
                            json={
                                "policy_number": policy_number,
                                "query": prompt,
                                "session_id": st.session_state.session_id
                            },
                            headers=headers,
                            timeout=60
                        )
                        if res.status_code == 200:
                            response_content = res.json().get("message", "No response from backend.")
                        elif res.status_code == 401:
                            response_content = "⛔ Your session has expired. Please log in again."
                            st.session_state.jwt_token = None
                            st.session_state.user_info = None
                        else:
                            response_content = f"🚫 API Error: {res.status_code} - {res.text}"
                    except Exception:
                        response_content = "🚫 Unable to reach the backend service. Please try again later."

                with st.chat_message("assistant"):
                    st.markdown(response_content)
                st.session_state.messages.append({"role": "assistant", "content": response_content})

if __name__ == "__main__":
    InsuranceVirtualAgentPage().render()
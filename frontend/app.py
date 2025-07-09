import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import uuid
from src.logging.logger import logging
import requests
import streamlit as st
from src.workflow.graph import Graph
from src.exception.exception import InsuranceAgentException
from src.constant import RAG_ENDPOINTS


class InsuranceVirtualAgentApp:
    def __init__(self):
        """
        Initialize session id and chat history.
        """
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        self.session_id = st.session_state.session_id

        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hi, I'm your insurance policy agent. How may I help you?"
                }
            ]

    def render_policy(self):
        """
        Render policy number input and validate.
        """
        st.set_page_config(page_title="Insurance Virtual Agent", layout="centered", page_icon="🛡️")
        st.title("🏦 Insurance Virtual Agent")
        st.markdown("Please enter your policy number to continue.")

        policy_number = st.text_input(
            "🔢 Enter your Policy Number",
            placeholder="e.g., AU1234",
            max_chars=6
        )
        if policy_number:
            policy_number = policy_number.strip().upper()
            if self.validate_policy(policy_number):
                st.success("Policy number validated.")
                return policy_number
            else:
                st.error("❌ Invalid policy number. Please enter a valid policy number (e.g., AU1234).")
        return None

    @staticmethod
    def validate_policy(policy_number):
        """
        Validate policy number format.
        """
        return bool(re.fullmatch(r"AU\d{4}", policy_number))

    def render_ui(self):
        """
        Render chat UI and handle user input.
        """
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
                return None
            if len(prompt) > 500:
                st.warning("Query too long. Please keep it under 500 characters.")
                return None

            # Display user message and add to history
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            return prompt
        return None

    def process_query(self, user_input, policy_number):
        """
        Send user query to RAG backend and display response.
        """
        try:
            with st.spinner("Thinking..."):
                try:
                    res = requests.post(
                        RAG_ENDPOINTS,
                        json={
                            "policy_number": policy_number,
                            "query": user_input,
                            "session_id": self.session_id
                        },
                        timeout=60  # Set a timeout for the request
                    )
                except requests.RequestException as e:
                    logging.error(f"API request failed: {e}")
                    response_content = "🚫 Unable to reach the backend service. Please try again later."
                else:
                    if res.status_code == 200:
                        response_content = res.json().get("message", "No response from backend.")
                    else:
                        logging.error(f"API error {res.status_code}: {res.text}")
                        response_content = f"🚫 API Error: {res.status_code} - {res.text}"

                # Display assistant response and add to history
                with st.chat_message("assistant"):
                    st.markdown(response_content)
                st.session_state.messages.append({"role": "assistant", "content": response_content})

        except Exception as e:
            logging.exception("Unexpected error in process_query")
            st.error("An unexpected error occurred. Please try again later.")
            raise InsuranceAgentException(e, sys)

    def main(self):
        """
        Main function to execute the entire application.
        """
        try:
            policy_number = self.render_policy()
            if policy_number:
                user_input = self.render_ui()
                if user_input:
                    self.process_query(user_input, policy_number)
        except Exception as e:
            logging.exception("Error in main application loop")
            st.error("A critical error occurred. Please contact support.")
            raise InsuranceAgentException(e, sys)

# This code is the main entry point for the Insurance Virtual Agent application.
# It initializes the application, renders the UI for policy number input,
# and handles user queries.
if __name__ == "__main__":
    app_obj = InsuranceVirtualAgentApp()
    app_obj.main()

import os
import re, sys
import streamlit as st
import uuid
from src.workflow.graph import Graph
from src.exception.exception import InsuranceAgentException
from langchain_core.messages import HumanMessage


class InsuranceVirtualAgentApp():
    def __init__(self):
        """ 
        Initialize Graph class if not in session state
        """
        if "graph" not in st.session_state:
            st.session_state.graph = Graph()
        if "thread_id" not in st.session_state:
            st.session_state.thread_id = str(uuid.uuid4())

    def render_policy(self):
        try:
            
            st.set_page_config(page_title="Insurance Virtual Agent", layout="centered")
            st.title("🏦 Insurance Virtual Agent")

            policy_number = st.text_input("Enter your Policy Number")
            if policy_number:
                return policy_number
        except Exception as e:
            raise InsuranceAgentException(e,sys)
    
    def validate_policy(self,policy_number):
        try:
            return bool(re.match(r'^AU\d{4}$', policy_number))
        except Exception as e:
            raise InsuranceAgentException(e,sys)
    
    def render_ui(self):
        try:
            #Initialize chat history 
            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {"role":"assistant",
                    "content":"Hi, I'm your insurance policy agent. How may I help you?"}
                ]
            #Display chat messages from history on app rerun
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            #Accept User input 
            if prompt:=st.chat_input(placeholder="Enter your query related to policy"):
                #Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)
                #Add user message to the chat history
                st.session_state.messages.append({"role":"user","content":prompt})

                return prompt
        
        except Exception as e:
            raise InsuranceAgentException(e,sys)


    def process_query(self,user_input,policy_number):
        try:
            message = [HumanMessage(content=user_input)]
            config = {"configurable":{"thread_id":st.session_state.thread_id}}
            graph = st.session_state.graph.build_graph()
            response = graph.invoke({"messages":message,"policy_number":policy_number},config)
            response_content = response["messages"][-1].content
            #Display assistant response in chat message container
            with st.chat_message("assistant"):
            
                st.session_state.messages.append({"role":"assistant","content":response_content})
                st.markdown(response_content)

        except Exception as e:
            raise InsuranceAgentException(e,sys)

    def main(self):
        """ 
        Main function to execute entire application
        """
        try:
            policy_number = self.render_policy()

            if policy_number:
                if self.validate_policy(policy_number):
                    user_input = self.render_ui()
                    if user_input:
                        self.process_query(user_input,policy_number)
                    else:
                        st.write("Please enter the query to proceed")
                else:
                    st.write("Policy Number invalid. Please enter valid policy number")
        
        except Exception as e:
            raise InsuranceAgentException(e,sys)
        
app_obj = InsuranceVirtualAgentApp()
app_obj.main()



    



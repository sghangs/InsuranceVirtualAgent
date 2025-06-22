import os
import sys
from typing import TypedDict, Annotated,List, Dict, Literal
from functools import partial
from pydantic import BaseModel, Field
import operator
import datetime
import uuid
import ast
from IPython.display import Image, display
import boto3
import json

#langchain imports
from langchain.prompts import PromptTemplate
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    AIMessage,
    RemoveMessage
)
from langchain_core.documents import Document
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

#langgraph imports
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode,tools_condition

from pinecone import Pinecone

#Project package imports
from src.retriever.retriever import HybridRetriever
from src.llm import llm
from src.constant import PINECONE_INDEX_NAME
from src.exception.exception import InsuranceAgentException
from src.logging.logger import logging
from src.constant import MESSAGES_COUNT
from src.prompts.prompts import (
    generate_input_prompt,
    generate_response_chain,
    generate_rewrite_chain
)
from src.prompts.graders import (
    grade_answer,
    grade_hallucinations,
    grade_documents
)

#load environment variables
from dotenv import load_dotenv
load_dotenv()


# State schema for graph
class State(MessagesState):
    policy_number : str
    filtered_docs : List[str]
    summary : str


class Graph():
    def __init__(self) -> None:
        """
        Initialize retriever and bind as a tool to llm
        """
        try:
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            index_name = PINECONE_INDEX_NAME

            self.retriever = HybridRetriever(pinecone_api_key,index_name)
            self.tools = self.retriever.get_tools()
            self.llm_with_tools = llm.bind_tools(self.tools)
            logging.info("Retriever Initialization and bind tools completed")

        except Exception as e:
            raise InsuranceAgentException(e,sys)


    def summarize_conversation(self,State):
        """
        summarize the converstations 
        """
        logging.info("Entering into sumarize_conversation...")
        try:
            #First, we get any existing summary
            summary = State.get("summary", "")

            #create summarization prompt
            if summary:
                #A summary already exists
                summary_message = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
                )
            else:
                summary_message = "Create a summary of the conversation above:"

            # Add prompt to our history
            conversation_messages = [message for message in State["messages"] if message.type in ("human","system")
                                        or (message.type == "ai" and not message.tool_calls)]

            messages = conversation_messages + [HumanMessage(content=summary_message)]
            response = llm.invoke(messages)

            # Delete all but keep the one most recent messages
            delete_messages = [RemoveMessage(id=m.id) for m in conversation_messages[:-1]]

            return {"summary": response.content, "messages": delete_messages}
        
        except Exception as e:
            raise InsuranceAgentException(e,sys)

    def should_continue(self,State) -> Literal["summarize_conversation","generate_toolcall_or_respond"]:
        """
        Return the next node to execute.
        """
        try:
    
            messages = State["messages"]

            conversation_messages = [message for message in State["messages"] if message.type in ("human","system")
                                        or (message.type == "ai" and not message.tool_calls)]
            
            # If there are more than given messages, then we summarize the conversation
            if len(conversation_messages) > MESSAGES_COUNT:
                return "summarize_conversation"
            
            # Otherwise we can skip summarization
            return "generate_toolcall_or_respond"
        
        except Exception as e:
            raise InsuranceAgentException(e,sys)

     
    def generate_toolcall_or_respond(self,State):
        """ 
        Generate tool call for the retriever tool based on the user query and policy number.
        or respond to the user directly if the query is not related to any policy.
        """
        logging.info("Entering into generate_toolcall_or_respond...")
        try:
            # Get summary if it exists
            summary = State.get("summary", "")

            conversation_messages = [message for message in State["messages"] if message.type in ("human","system")
                                    or (message.type == "ai" and not message.tool_calls)]
            if summary:
                # Add summary to system message
                system_message = f"Summary of conversation earlier: {summary}"

                # Append summary to any newer messages
                messages = [SystemMessage(content=system_message)] + conversation_messages
            else:
                messages = conversation_messages

            input_prompt = generate_input_prompt()
            prompt = input_prompt.invoke({
                "question": messages,
                "policy_number": State["policy_number"]
            })
            response = self.llm_with_tools.invoke(prompt)
            
            return {"messages": [response]}

        except Exception as e:
            raise InsuranceAgentException(e,sys)

    
    def grade_documents(self,State): 
        """ 
        filter the retrieved documents based on their relevance to the query 
        """
        logging.info("Entering grade_documents...")
        try:
            # Get summary if it exists
            summary = State.get("summary", "")

            # Get the converstation messages
            conversation_messages = [message for message in State["messages"] if message.type in ("human","system")
                                    or (message.type == "ai" and not message.tool_calls)]
            
            if summary:
                # Add summary to system message
                system_message = f"Summary of conversation earlier: {summary}"

                # Append summary to any newer messages
                messages = [SystemMessage(content=system_message)] + conversation_messages
            else:
                messages = conversation_messages

            # Get the retrieved documents from the last tool message
            recent_tool_messages = []
            for msg in reversed(State["messages"]):
                if msg.type == "tool":
                    recent_tool_messages.append(msg)
                else:
                    break
            tool_messages = recent_tool_messages[::-1]

            # convert from string "[]" into [] (list)
            docs_list = ast.literal_eval(tool_messages[0].content)

            relevant_docs = []
            retrieval_grader = grade_documents()

            for doc in docs_list:
                score = retrieval_grader.invoke({"question": messages, "document": doc["metadata"]["text"]})
                grade = score.binary_score
                if grade not in ["yes", "no"]:
                    raise ValueError(f"Invalid score received: {grade}. Expected 'yes' or 'no'.")
                if grade == "yes":
                    relevant_docs.append(doc['metadata']['text'])

            if relevant_docs:
                return {"filtered_docs":relevant_docs}
            else:
                return {"filtered_docs": []}
            
        except Exception as e:
            raise InsuranceAgentException(e,sys)
    
    def decide_to_generate(self,State) -> Literal["generate_answer", "no_relevant_documents"]:
        """ 
        Decide which node to execute next based on the retrieved documents.
        If no documents are retrieved, respond directly to the user.
        """
        try:
            if not State["filtered_docs"]:
                return "rewrite_query"
            else:
                return "generate_answer"
            
        except Exception as e:
            raise InsuranceAgentException(e,sys)

    def generate_answer(self,State):
        """ 
        Generate the response based on retrieved documents and query for that given
        policy
        """
        logging.info("Entering generate_answer...")
        try:
            # Get summary if it exists
            summary = State.get("summary", "")

            context = "/n/n".join(doc for doc in State["filtered_docs"])

            # Get the converstation messages
            conversation_messages = [message for message in State["messages"] if message.type in ("human","system")
                                    or (message.type == "ai" and not message.tool_calls)]
            
            if summary:
                # Add summary to system message
                system_message = f"Summary of conversation earlier: {summary}"

                # Append summary to any newer messages
                messages = [SystemMessage(content=system_message)] + conversation_messages
            else:
                messages = conversation_messages
            
            generation_chain = generate_response_chain()
            response = generation_chain.invoke({
                "context": context,
                "question": messages
            })
            return {"messages":[response]}
        
        except Exception as e:
            raise InsuranceAgentException(e,sys)
    
    def rewrite_query(self,State):
        """ 
        Rewrite the query if no relevant documents are found.
        """
        logging.info("Entering rewrite_query...")
        try:
            for message in reversed(State["messages"]):
                if message.type == "human":
                    question = message.content
                    break
            
            rewrite_chain = generate_rewrite_chain()
            response = rewrite_chain.invoke({
                "question": question
            })
            rewritten_query = response.content
            print(f"Rewritten query: {rewritten_query}")
        
            return {"messages": [{"role": "human", "content": rewritten_query}]}
        
        except Exception as e:
            raise InsuranceAgentException(e,sys)
    
    def decide_to_regenerate(self,State) -> Literal["useful", "not supported"]:
        """ 
        Decide whether to regenerate the answer based on hallucination and answer grading.
        If the answer is not grounded in the retrieved documents, it is considered "not supported".
        """
        try:
            # Get summary if it exists
            summary = State.get("summary", "")
            message = State["messages"][-1]
            
            context = "/n/n".join(doc for doc in State["filtered_docs"])

            for msg in reversed(State["messages"]):
                if msg.type == "human":
                    question = msg
                    # If the last message is a human message, use it as the question
                    break

            if summary:
                # Add summary to system message
                system_message = f"Summary of conversation earlier: {summary}"

                # Append summary to any newer messages
                question_with_summary = [SystemMessage(content=system_message)] + [question]
            else:
                question_with_summary = question

            hallucination_grader = grade_hallucinations()
            score = hallucination_grader.invoke({
                "documents": context,
                "generation": message.content
            })
            grade = score.binary_score
            if grade not in ["yes", "no"]:
                raise ValueError(f"Invalid score received: {grade}. Expected 'yes' or 'no'.")
            
            answer_grader = grade_answer()
            if grade == "yes":
                response = answer_grader.invoke({
                    "question": question_with_summary,
                    "generation": message.content
                })
                grade = response.binary_score
                if grade not in ["yes", "no"]:
                    raise ValueError(f"Invalid score received: {grade}. Expected 'yes' or 'no'.")
                if grade == "yes":
                    return "useful"
                else:
                    return "not supported"
            else:     
                return "not supported"   
        
        except Exception as e:
            raise InsuranceAgentException(e,sys)
      

    def build_graph(self):
        """ 
        Build the graph using defined nodes
        """
        try:
            workflow = StateGraph(State)

            #Add nodes to the graph
            workflow.add_node("summarize_conversation",self.summarize_conversation)
            workflow.add_node("generate_toolcall_or_respond",self.generate_toolcall_or_respond)
            workflow.add_node("retrieve", ToolNode(self.tools))
            workflow.add_node("grade_documents",self.grade_documents)
            workflow.add_node("generate_answer",self.generate_answer)
            workflow.add_node("rewrite_query",self.rewrite_query)
            
            #Define edges for workflow
            workflow.add_conditional_edges(
                START,self.should_continue,
                {
                    "summarize_conversation","summarize_conversation",
                    "generate_toolcall_or_respond","generate_toolcall_or_respond"
                }
            )
            workflow.add_edge("summarize_conversation","generate_toolcall_or_respond")
            workflow.add_conditional_edges(
                "generate_toolcall_or_respond",tools_condition,
                {
                    "tools":"retrieve",
                    END:END
                }
            )
            workflow.add_edge("retrieve","grade_documents")
            workflow.add_conditional_edges(
                "grade_documents", self.decide_to_generate,
                {
                    "generate_answer": "generate_answer",
                    "rewrite_query": "rewrite_query"
                }
            )
            workflow.add_conditional_edges(
                "generate_answer", self.decide_to_regenerate,
                {
                    "useful": END,
                    "not supported": "generate_answer"
                }
            )
            workflow.add_edge("rewrite_query", "generate_toolcall_or_respond")

            memory = MemorySaver()
            graph = workflow.compile(checkpointer=memory)

            return graph
        
        except Exception as e:
            raise InsuranceAgentException(e,sys)




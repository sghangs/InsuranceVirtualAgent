import os
import sys
from typing import TypedDict, Annotated,List, Dict, Literal
from functools import partial
from pydantic import BaseModel, Field
import operator
import datetime
import uuid
import ast
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
from langchain_core.runnables.config import RunnableConfig

#langgraph imports
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.checkpoint.redis import AsyncRedisSaver


from pinecone import Pinecone

#Project package imports
from src.retriever.retriever import HybridRetriever
from src.llm import llm
from src.constant import PINECONE_DENSE_INDEX_NAME,PINECONE_SPARSE_INDEX_NAME
from src.exception.exception import InsuranceAgentException
from src.loggers.logger import logging
from src.constant import MESSAGES_COUNT
from src.prompts.prompts import (
    generate_input_prompt,
    generate_response_chain,
    generate_rewrite_chain,
    user_profile_prompt
)
from src.prompts.graders import (
    grade_answer,
    grade_hallucinations,
    grade_documents
)

from src.longmemory.profile import get_user_profile
from src.longmemory.embedding_utility import should_embed,get_embedding
from src.longmemory.memory import retrieve_similar_conversations,store_summary
from src.longmemory.profile_extraction import extract_profile_updates
from src.longmemory.profile import upsert_user_profile
from src.shortmemory.config import REDIS_URL
from src.shortmemory.memory import RedisMemoryManager
from src.shortmemory.redis_checkpointer import RedisCheckpointer
#load environment variables
from dotenv import load_dotenv
load_dotenv()


# State schema for graph
class State(MessagesState):
    policy_number : Annotated[str, Field(description="Insurance policy number associated with the user")]
    filtered_docs : Annotated[List[Document], Field(description="List of filtered documents relevant to the user query")]
    summary : Annotated[str, Field(description="Summary of the conversation")]
    user_id : Annotated[str, Field(description="Unique identifier for the user")]
    user_profile : Annotated[Dict[str, str], Field(description="User profile information")]
    long_term_context : Annotated[str, Field(description="Long-term context for the user")]


class Graph():
    def __init__(self) -> None:
        """
        Initialize retriever and bind as a tool to llm
        """
        try:
            pinecone_api_key = os.getenv("PINECONE_API_KEY")        
            dense_index_name = PINECONE_DENSE_INDEX_NAME
            sparse_index_name = PINECONE_SPARSE_INDEX_NAME

            self.retriever = HybridRetriever(pinecone_api_key,dense_index_name,sparse_index_name)
            self.tools = self.retriever.get_tools()
            self.llm_with_tools = llm.bind_tools(self.tools)
            logging.info("Retriever Initialization and bind tools completed")

        except Exception as e:
            raise InsuranceAgentException(e,sys)


    async def summarize_conversation(self,State,config: RunnableConfig) -> Dict[str, List[AnyMessage]]:
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
            print("Response from summarization:", response.content)

            thread_id = config["configurable"]["thread_id"]
            # Store the summary as long term memory in the database
            await store_summary(
                policy_number=State["policy_number"],
                user_id=State["user_id"],
                summary=response.content,
                thread_id=thread_id
            )

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


    async def generate_toolcall_or_respond(self,State: dict, config: RunnableConfig) -> Dict[str, List[AnyMessage]]:
        """ 
        Generate tool call for the retriever tool based on the user query and policy number.
        or respond to the user directly if the query is not related to any policy.
        """
        logging.info("Entering into generate_toolcall_or_respond...")
        try:
            # Get user_id from the config
            State["user_id"] = config["configurable"]["user_id"]
            #extract profile updates from the last message    
            profile_updates = await extract_profile_updates(State["messages"][-1].content)
            print("Profile updates extracted:", profile_updates)
            if profile_updates:
                # Update user profile with the extracted information
                await upsert_user_profile(State["user_id"], profile_updates)

            # Get user profile information
            State["user_profile"] = await get_user_profile(State["user_id"])
        

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

            # Get the embedding for the last user message if it is not trivial
            if should_embed(messages[-1].content):
                # retreive similar conversations of the user based on the last message and policy number    
                long_term_conversations= await retrieve_similar_conversations(State["user_id"], messages[-1].content, State["policy_number"])
                State["long_term_context"] = "\n".join([f"{msg['user_id']} ({msg['created_at']}): {msg['message']}" for msg in long_term_conversations])
            else:
                State["long_term_context"] = ""

            # Get user profile information as a string
            user_profile_info = user_profile_prompt(State.get("user_profile", {}))

            # Generate input prompt and get the response
            # If the question is related to the insurance policy, make a tool call to the retriever
            input_prompt = generate_input_prompt()
            prompt = input_prompt.invoke({
                "question": messages,
                "policy_number": State["policy_number"],
                "user_profile_info": user_profile_info,
                "long_term_context": State["long_term_context"]
            })
            response = await self.llm_with_tools.ainvoke(prompt)
            

            return {"messages": [response],"filtered_docs":[], "summary": summary,
                    "policy_number": State["policy_number"],
                    "user_id": State["user_id"],
                    "user_profile": State["user_profile"],
                    "long_term_context": State["long_term_context"]}

        except Exception as e:
            raise InsuranceAgentException(e,sys)

    
    async def grade_documents(self,State): 
        """ 
        filter the retrieved documents based on their relevance to the query 
        """
        logging.info("Entering grade_documents...")
        print("Printing state grade_documents:", State["user_profile"])
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
                score = await retrieval_grader.ainvoke({"question": messages, "document": doc})
                grade = score.binary_score
                if grade not in ["yes", "no"]:
                    raise ValueError(f"Invalid score received: {grade}. Expected 'yes' or 'no'.")
                if grade == "yes":
                    relevant_docs.append(doc)
            
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

    async def generate_answer(self,State):
        """ 
        Generate the response based on retrieved documents and query for that given
        policy
        """
        logging.info("Entering generate_answer...")
        print("Printing state generate_answer:", State["user_profile"])
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
            
            
            # Get user profile information as a string
            user_profile_info = user_profile_prompt(State.get("user_profile"))

            generation_chain = generate_response_chain()
            response = await generation_chain.ainvoke({
                "context": context,
                "question": messages,
                "user_profile_info": user_profile_info,
                "long_term_context": State["long_term_context"]
            })
            return {"messages":[response],"filtered_docs":State["filtered_docs"]}
        
        except Exception as e:
            raise InsuranceAgentException(e,sys)
    
    async def rewrite_query(self,State):
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
            response = await rewrite_chain.ainvoke({
                "question": question
            })
            rewritten_query = response.content
            print(f"Rewritten query: {rewritten_query}")
        
            return {"messages": [{"role": "human", "content": rewritten_query}]}
        
        except Exception as e:
            raise InsuranceAgentException(e,sys)
    
    async def decide_to_regenerate(self,State) -> Literal["useful", "not supported"]:
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
            score = await hallucination_grader.ainvoke({
                "documents": context,
                "generation": message.content
            })
            grade = score.binary_score
            if grade not in ["yes", "no"]:
                raise ValueError(f"Invalid score received: {grade}. Expected 'yes' or 'no'.")
            
            answer_grader = grade_answer()
            if grade == "yes":
                response = await answer_grader.ainvoke({
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
      

    async def build_graph(self):
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


            # Create the Redis checkpointer
            #redis_memory = RedisMemoryManager()
            #checkpointer = RedisCheckpointer(redis_memory)
            ttl_config = {
                "default_ttl": 60,  # Default TTL in minutes
                "refresh_on_read": True,  # Refresh TTL when store entries are read
        }
            async with AsyncRedisSaver.from_conn_string(
                redis_url=REDIS_URL,
                ttl=ttl_config  # Set TTL for the checkpointer
            ) as checkpointer:
                await checkpointer.asetup()
                graph = workflow.compile(checkpointer=checkpointer)

                return graph
        
        except Exception as e:
            raise InsuranceAgentException(e,sys)




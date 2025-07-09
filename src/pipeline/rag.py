from src.workflow.graph import Graph
import uuid
import sys
from langchain_core.messages import HumanMessage
from src.logging.logger import logging
from src.exception.exception import InsuranceAgentException


class RagPipeline():
    """
    Implement RAG pipeline to extract the retrieved documents and response.
    """

    def __init__(self):
        """
        Initialize graph object and build the graph.
        This graph will be used to execute the RAG application.
        """
        self.graph_obj=Graph()
        self.graph = self.graph_obj.build_graph()

    async def execute_rag(self,user_input,policy_number,session_id):
        """
        execute the rag application with provided input to return the response and context
        """
        try:
            message = [HumanMessage(content=user_input)]
            config = {"configurable":{"thread_id":session_id}}
            
            response = self.graph.invoke({"messages":message,"policy_number":policy_number},config)
            response_content = response["messages"][-1].content
            context = response["filtered_docs"]

            return response_content,context
        except Exception as e:
            raise InsuranceAgentException(sys,e)
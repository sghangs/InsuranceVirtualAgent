from src.workflow.graph import Graph
import uuid
import sys
from langchain_core.messages import HumanMessage
from src.loggers.logger import logging
from src.exception.exception import InsuranceAgentException


class RagPipeline():
    """
    Implement RAG pipeline to extract the retrieved documents and response.
    """

    def __init__(self, graph_obj, graph):
        """
        Initialize graph object and build the graph.
        """
        self.graph_obj = graph_obj
        self.graph = graph

    @classmethod
    async def create(cls):
        """
        Async factory to initialize RagPipeline with an async build_graph.
        """
        graph_obj = Graph()
        graph = await graph_obj.build_graph()
        return cls(graph_obj, graph)

    async def execute_rag(self,user_input,policy_number,session_id,user_id):
        """
        execute the rag application with provided input to return the response and context
        """
        try:
            message = [HumanMessage(content=user_input)]
            config = {"configurable":{"thread_id":session_id,"user_id":user_id}}
            
            response = await self.graph.ainvoke({"messages":message,"policy_number":policy_number},config)
            response_content = response["messages"][-1].content
            context = response["filtered_docs"]

            return response_content,context
        except Exception as e:
            raise InsuranceAgentException(sys,e)
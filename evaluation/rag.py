from src.workflow.graph import Graph
import uuid
from langchain_core.messages import HumanMessage


class RagPipeline():
    """
    Implement RAG pipeline to extract the retrieved documents and response.
    """

    def __init__(self):
        """
        Initialize graph object, thread id and policy number
        """
        self.graph_obj=Graph()
        self.thread_id = str(uuid.uuid4())
        self.policy_number = "AU1234"


    def execute_rag(self,user_input):
        """
        execute the rag application with provided input to return the response and context
        """
        message = [HumanMessage(content=user_input)]
        config = {"configurable":{"thread_id":self.thread_id}}
        graph = self.graph_obj.build_graph()
        response = graph.invoke({"messages":message,"policy_number":self.policy_number},config)
        response_content = response["messages"][-1].content
        context = response["filtered_docs"]

        return response_content,context
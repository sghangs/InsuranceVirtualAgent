from src.workflow.graph import Graph
import uuid
from langchain_core.messages import HumanMessage



graph_obj = Graph()

thread_id = str(uuid.uuid4())
policy_number = "AU1234"

def execute_rag(user_input,policy_number="AU1234",thread_id=str(uuid.uuid4())):
    message = [HumanMessage(content=user_input)]
    config = {"configurable":{"thread_id":thread_id}}
    graph = graph_obj.build_graph()
    response = graph.invoke({"messages":message,"policy_number":policy_number},config)
    response_content = response["messages"][-1].content

    return response_content
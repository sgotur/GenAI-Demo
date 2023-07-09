# diagram.py
from diagrams import Cluster, Diagram, Edge
from diagrams.custom import Custom
from diagrams.aws.ml import Comprehend
from diagrams.aws.ml import SagemakerModel
from diagrams.generic.blank import Blank
from diagrams.aws.storage import S3
from diagrams.aws.general import Users
from diagrams.programming.language import Python

with Diagram("GenAI_geospatial_analyzer_architecture", show=False):
    users = Users("Demo Users")
    demo_app = Python("Streamlit Python App")

    with Cluster("PII"):
        comprehend = Comprehend("Amazon Comprehend detect and obfuscate PII")
    
    with Cluster("Maps"):
        location_service = Custom("Amazon Location Service", "./amzn-loc-icon.jpg")


    with Cluster("Summarization"):
        model_process1 = SagemakerModel("Generative AI for content summarization")
        model_process2 = SagemakerModel("Generative AI auto-suggest additional prompts")
        model_process4 = SagemakerModel("Generative AI question answering")
                                                                 

    users >> Edge(minlen="2") >> demo_app
    demo_app >> Edge(minlen="4") >> location_service
    demo_app << Edge(minlen="4") << location_service
    demo_app >> Edge(minlen="4") >> model_process1
    model_process1 >> Edge(minlen="4") >> model_process2
    demo_app << Edge(minlen="2") << model_process1
    demo_app >> Edge(minlen="4") >> comprehend >> Edge(minlen="4") >> model_process4
    model_process4 >> Edge(minlen="4") >> model_process2
    demo_app << Edge(minlen="4") << model_process4
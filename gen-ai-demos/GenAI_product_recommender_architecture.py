# diagram.py
from diagrams import Cluster, Diagram, Edge
from diagrams.custom import Custom
from diagrams.aws.ml import Comprehend
from diagrams.aws.ml import Personalize
from diagrams.aws.ml import SagemakerModel
from diagrams.generic.blank import Blank
from diagrams.aws.storage import S3
from diagrams.aws.general import Users
from diagrams.programming.language import Python

with Diagram("GenAI_product_recommender_architecture", show=False):
    users = Users("Demo Users")
    demo_app = Python("Streamlit Python App")

    with Cluster("Knowledge management"):
        personalize = Personalize("Amazon Personalize for similar items recommendation")
        comprehend = Comprehend("Amazon Comprehend to detect and obfuscate PII")
    
    with Cluster("Summarization"):
        model_process2 = SagemakerModel("Generative AI auto-suggest additional prompts")
        model_process4 = SagemakerModel("Generative AI question answering")
                                                                 

    users >> Edge(minlen="2") >> demo_app
    demo_app >> Edge(minlen="4") >> personalize
    demo_app << Edge(minlen="4") << personalize
    demo_app >> Edge(minlen="4") >> comprehend >> Edge(minlen="4") >> model_process4
    model_process4 >> Edge(minlen="4") >> model_process2
    demo_app << Edge(minlen="4") << model_process4
# diagram.py
from diagrams import Cluster, Diagram, Edge
from diagrams.custom import Custom
from diagrams.aws.ml import Comprehend
from diagrams.aws.ml import SagemakerModel
from diagrams.generic.blank import Blank
from diagrams.aws.general import Users
from diagrams.programming.language import Python

with Diagram("GenAI_product_ideator_architecture", show=False):
    users = Users("Demo Users")
    demo_app = Python("Streamlit Python App - enter product idea in a few words")

    with Cluster("image generation"):
        stable = SagemakerModel("SageMaker Jumpstart Stable diffusion model for text to image")
        
    
    with Cluster("content generation"):
        comprehend = Comprehend("Amazon Comprehend to suggest key product pointers")
        model_process2 = SagemakerModel("Generative AI product description")
        model_process3 = SagemakerModel("Generative AI press release")
                                                                 

    users >> Edge(minlen="4") >> demo_app
    demo_app >> Edge(minlen="6") >> stable
    demo_app << Edge(minlen="6") << stable
    demo_app >> Edge(minlen="6") >> model_process2
    demo_app << Edge(minlen="6") << model_process2
    demo_app << Edge(minlen="6") << comprehend
    demo_app >> Edge(minlen="6") >> model_process3
    demo_app << Edge(minlen="6") << model_process3

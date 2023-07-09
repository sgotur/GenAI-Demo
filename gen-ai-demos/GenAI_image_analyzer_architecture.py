# diagram.py
from diagrams import Cluster, Diagram, Edge
from diagrams.aws.ml import Comprehend
from diagrams.aws.ml import Rekognition
from diagrams.aws.ml import SagemakerModel
from diagrams.generic.blank import Blank
from diagrams.aws.storage import S3
from diagrams.aws.general import Users
from diagrams.programming.language import Python

with Diagram("GenAI_image_analyzer_architecture", show=False):
    users = Users("Demo Users")
    demo_app = Python("Streamlit Python App")
    image_extraction = Rekognition("Detect labels from images")

    with Cluster("Upload content"):
        upload_file = S3("Amazon S3 bucket storage")
        download_file = S3("Amazon S3 bucket storage")
        comprehend = Comprehend("Amazon Comprehend detect and obfuscate PII")
    
    with Cluster("Summarization"):
        model_process1 = SagemakerModel("Generative AI for content summarization")
        model_process2 = SagemakerModel("Generative AI auto-suggest additional prompts")
        model_process4 = SagemakerModel("Generative AI question answering")
                                                                 

    users >> Edge(minlen="2") >> demo_app >> Edge(minlen="2") >> upload_file
    download_file >> Edge(minlen="4") >> image_extraction
    image_extraction >> Edge(minlen="4") >> model_process1
    model_process1 >> Edge(minlen="4") >> model_process2
    demo_app << Edge(minlen="2") << model_process1
    demo_app >> Edge(minlen="4") >> comprehend >> Edge(minlen="4") >> model_process4
    model_process4 >> Edge(minlen="4") >> model_process2
    demo_app << Edge(minlen="4") << model_process4
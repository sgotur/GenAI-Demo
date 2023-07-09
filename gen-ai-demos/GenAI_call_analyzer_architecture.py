# diagram.py
from diagrams import Cluster, Diagram, Edge
from diagrams.aws.ml import Comprehend
from diagrams.aws.ml import Transcribe
from diagrams.aws.ml import SagemakerModel
from diagrams.generic.blank import Blank
from diagrams.aws.storage import S3
from diagrams.aws.general import Users
from diagrams.programming.language import Python

with Diagram("GenAI_call_analyzer_architecture", show=False):
    users = Users("Demo Users")
    demo_app = Python("Streamlit Python App")

    with Cluster("Transcribe call"):
        load_file = S3("Amazon S3 bucket storage")
        transcribe = Transcribe("Amazon Transcribe call analytics")
        comprehend = Comprehend("Amazon Comprehend detect and obfuscate PII")
    
    with Cluster("Summarization"):
        model_process1 = SagemakerModel("Generative AI transcript summarization")
        model_process2 = SagemakerModel("Generative AI auto-suggest additional prompts")
        model_process3 = SagemakerModel("Generative AI neural translation")
        model_process4 = SagemakerModel("Generative AI question answering")
                                                                 

    #users >> Edge(label="interact with") >> demo_app >> Edge(label="Upload audio file") >> load_file
    #load_file >> Edge(label="Submit call analytics job") >> transcribe >> Edge(label="Send call analytics and transcript to Amazon Comprehend") >> comprehend
    #comprehend >> Edge(label="Send PII masked transcript") >> model_process >> Edge(label="Send transcript summary and display in app") >> demo_app

    users >> Edge(minlen="2") >> demo_app >> Edge(minlen="2") >> load_file
    load_file >> Edge(minlen="4") >> transcribe >> Edge(minlen="4") >> comprehend
    comprehend >> Edge(minlen="4") >> model_process1
    model_process1 >> Edge(minlen="4") >> model_process2 >> Edge(minlen="4") >> model_process3
    demo_app << Edge(minlen="2") << model_process1
    demo_app >> Edge(minlen="4") >> comprehend >> Edge(minlen="2") >> model_process4
    model_process4 >> Edge(minlen="4") >> model_process2 >> Edge(minlen="4") >> model_process3 


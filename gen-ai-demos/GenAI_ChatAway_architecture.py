# diagram.py
from diagrams import Cluster, Diagram, Edge
from diagrams.aws.ml import Comprehend
from diagrams.aws.ml import Transcribe
from diagrams.aws.ml import SagemakerModel
from diagrams.generic.blank import Blank
from diagrams.aws.database import Dynamodb
from diagrams.aws.storage import S3
from diagrams.aws.general import Users
from diagrams.programming.language import Python

with Diagram("GenAI_ChatAway_architecture", show=False):
    users = Users("Demo Users")
    demo_app = Python("Streamlit Python App")
    dynamo1 = Dynamodb("Amazon DynamoDB - Store chats")
    dynamo2 = Dynamodb("Amazon DynamoDB - Retrieve chats")

    with Cluster("PII detection"):
        comprehend1 = Comprehend("Amazon Comprehend - Detect and obfuscate PII in input")
        comprehend2 = Comprehend("Amazon Comprehend - Detect and obfuscate PII in output")
    
    with Cluster("Conversational AI"):
        model_process1 = SagemakerModel("Generative AI for turn-by-turn conversation")
        model_process2 = SagemakerModel("Generative AI auto-suggest additional prompts")
                                                                 

    #users >> Edge(label="interact with") >> demo_app >> Edge(label="Upload audio file") >> load_file
    #load_file >> Edge(label="Submit call analytics job") >> transcribe >> Edge(label="Send call analytics and transcript to Amazon Comprehend") >> comprehend
    #comprehend >> Edge(label="Send PII masked transcript") >> model_process >> Edge(label="Send transcript summary and display in app") >> demo_app

    users >> Edge(minlen="2") >> demo_app
    demo_app >> Edge(minlen="2") >> comprehend1
    comprehend1 >> Edge(minlen="4") >> model_process1
    model_process1 >> Edge(minlen="4") >> model_process2
    comprehend2 << Edge(minlen="4") << model_process1
    demo_app << Edge(minlen="4") << comprehend2
    demo_app >> Edge(minlen="2") >> dynamo1
    demo_app << Edge(minlen="2") << dynamo2
# diagram.py
from diagrams import Cluster, Diagram, Edge
from diagrams.custom import Custom
from diagrams.aws.ml import Comprehend
from diagrams.aws.ml import Kendra
from diagrams.aws.ml import Lex
from diagrams.aws.ml import SagemakerModel
from diagrams.generic.blank import Blank
from diagrams.aws.storage import S3
from diagrams.aws.general import Users
from diagrams.programming.language import Python

with Diagram("GenAI_digital_persona_architecture", show=False):
    users = Users("Demo Users")
    demo_app = Python("Streamlit Python App")
    soul_machines = Custom("Soul Machines Digital Human with industry expertise", "./SM-circle-logo-hover.jpg")


    with Cluster("Conversational AI"):
        lex = Lex("Amazon Lex for contextual turn-by-turn chat experience")
        kendra = Kendra("Amazon Kendra for natural language search")
        comprehend = Comprehend("Amazon Comprehend to detect and obfuscate PII")
    
    with Cluster("Summarization"):
        model_process = SagemakerModel("Generative AI question answering")
                                                                 

    users >> Edge(minlen="2") >> demo_app
    demo_app >> Edge(minlen="2") >> soul_machines
    soul_machines >> Edge(minlen="6") >> lex
    lex >> Edge(minlen="8") >> kendra
    lex << Edge(minlen="8") << kendra
    lex >> Edge(minlen="6") >> model_process
    soul_machines << Edge(minlen="6") << comprehend << Edge(minlen="6") << lex 

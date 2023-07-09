import streamlit as st

st.set_page_config(
    page_title="AWS GenAI Demo Showcase",
    page_icon="crystall_ball",
)

st.markdown(
    """
    ### :red[Note] 
    - These demos are for informational purposes only and primarily for internal AWS consumption
    - These can be shown to customers under NDA. For showcasing at large events please reach out to rangarap@ or dabounds@ for inputs
    - Please review and comply with the [Generative AI Acceptable Use Policy](https://policy.a2z.com/docs/568686/publication)
    - Use these selection of [samples for playing with the demos](https://amazon.awsapps.com/workdocs/index.html#/folder/085a7d2cc912f998468435fdf7eab6e9bb09ae855acfb9b16aea59de7d547e21).  
    - The demos should not be considered as an actual prototype or working version of a proposed solution
    - Source code available in the [GitLab repo](https://gitlab.aws.dev/dabounds/gen-ai-demos).
    """)


st.write("# :green[Welcome to the AWS Generative AI Demo Showcase!]")

st.write("### :orange[Just in!!]")
st.markdown('''
    - **[GenAI Text to Image for Marketing](/GenAI_text_to_image_marketing):** Stable diffusion demo for generating marketing content \n
    - **[GenAI Text to Image for Media](/GenAI_text_to_image_media):** Stable diffusion demo for media use cases \n
    - **[GenAI Text to Image for Video Games](/GenAI_text_to_image_video_games):** Stable diffusion demo to generate video game avatars  \n
    - **[GenAI Sales Accelerator](/GenAI_Security_Advisor):** MVP version and still under testing, but upload your SFDC reports and get your 2X2 doc \n
    ''')


tab1, tab2, tab3 = st.tabs(["Industry", "Usecase", "Function"])

with tab1:
    industry = st.selectbox(
        '**Select an industry to begin**',
        ('Select...', 'Automotive', 'Financial Services', 'Healthcare', 'Hospitality', 'Life Sciences', 'Legal', 'Manufacturing', 'Energy', 'Retail', 'Technology', 'Transport', 'Fashion'))

    if industry == 'Automotive':
        st.markdown('''
            ### Available demos \n\n
            - **[GenAI Call Analyzer](/GenAI_call_analyzer):** Call transcription, analytics, summarization, auto-prompts and conversational insights using Amazon Transcribe and a LLM \n
            - **[GenAI ChatAway](/GenAI_ChatAway):** Conversational chat-style interaction using a LLM \n
            - **[GenAI Content Analyzer](/GenAI_content_analyzer):** Upload images, csv or text files, get a summary, auto-prompts and conversational insights using Amazom Rekognition and a LLM \n
            - **[GenAI Document Processor](/GenAI_document_processor):** Summarization of multi-page PDFs and conversational insights using Amazon Textract and a LLM \n
            - **[GenAI Digital Persona](/GenAI_digital_persona):** Talk to a 3D Avatar playing the role of an Automotive Specialist
            - **[GenAI Image Analyzer](/GenAI_image_analyzer):** Analysis of images, label detection and summarization \n
            - **[GenAI Video Analyzer](/GenAI_video_analyzer):** Analysis of videos, label detection and summarization \n
        ''')
    elif industry == 'Energy':
        st.markdown('''
            ### Available demos
            - **[GenAI Call Analyzer](/GenAI_call_analyzer):** Call transcription, analytics, summarization, auto-prompts and conversational insights using Amazon Transcribe and a LLM \n
            - **[GenAI ChatAway](/GenAI_ChatAway):** Conversational chat-style interaction using a LLM \n
            - **[GenAI Content Analyzer](/GenAI_content_analyzer):** Upload images, csv or text files, get a summary, auto-prompts and conversational insights using Amazom Rekognition and a LLM \n
            - **[GenAI Digital Persona](/GenAI_digital_persona):** Talk to a 3D Avatar playing the role of a Geoscientist
            - **[GenAI Energy Upstream Analyzer](/GenAI_energy_upstream_analyzer):** Well log interpretation of IoT data, analysis of wellsite images, auto-prompts and conversational insights using a LLM \n
            - **[GenAI Enterprise Search](/GenAI_enterprise_search_interpreter):** Intelligent enterprise search, auto-prompts and conversational interpretation using Amazon Kendra and a LLM \n
            - **[GenAI Image Analyzer](/GenAI_image_analyzer):** Analysis of images, label detection and summarization \n
            - **[GenAI Video Analyzer](/GenAI_video_analyzer):** Analysis of videos, label detection and summarization \n
        ''')   
    elif industry == 'Financial Services':
        st.markdown('''
            ### Available demos \n\n
            - **[GenAI Call Analyzer](/GenAI_call_analyzer):** Call transcription, analytics, summarization, auto-prompts and conversational insights using Amazon Transcribe and a LLM \n
            - **[GenAI ChatAway](/GenAI_ChatAway):** Conversational chat-style interaction using a LLM \n
            - **[GenAI Content Analyzer](/GenAI_content_analyzer):** Upload images, csv or text files, get a summary, auto-prompts and conversational insights using Amazom Rekognition and a LLM \n
            - **[GenAI Digital Persona](/GenAI_digital_persona):** Talk to a 3D Avatar playing the role of a Financial Analyst
            - **[GenAI Document Processor](/GenAI_document_processor):** Summarization of multi-page PDFs, auto-prompts and conversational insights using Amazon Textract and a LLM \n
            - **[GenAI Enterprise Search](/GenAI_enterprise_search_interpreter):** Intelligent enterprise search, auto-prompts and conversational interpretation using Amazon Kendra and a LLM \n
            - **[GenAI Image Analyzer](/GenAI_image_analyzer):** Analysis of images, label detection and summarization \n
            - **[GenAI Video Analyzer](/GenAI_video_analyzer):** Analysis of videos, label detection and summarization \n
        ''')
    elif industry == 'Healthcare':
        st.markdown('''
            ### Available demos \n\n
            - **[GenAI Call Analyzer](/GenAI_call_analyzer):** Call transcription, analytics, summarization, auto-prompts and conversational insights using Amazon Transcribe and a LLM \n
            - **[GenAI ChatAway](/GenAI_ChatAway):** Conversational chat-style interaction using a LLM \n
            - **[GenAI Content Analyzer](/GenAI_content_analyzer):** Upload images, MS Word, Excel, PPT, PDF, CSV, HTML, or text files, get a summary, auto-prompts and conversational insights using Amazom Rekognition and a LLM \n
            - **[GenAI Document Processor](/GenAI_document_processor):** Summarization of multi-page PDFs, auto-prompts and conversational insights using Amazon Textract and a LLM \n
            - **[GenAI Image Analyzer](/GenAI_image_analyzer):** Analysis of images, label detection and summarization \n
            - **[GenAI Video Analyzer](/GenAI_video_analyzer):** Analysis of videos, label detection and summarization \n
        ''')
    elif industry == 'Hospitality':
        st.markdown('''
            ### Available demos \n\n
            - **[GenAI Call Analyzer](/GenAI_call_analyzer):** Call transcription, analytics, summarization, auto-prompts and conversational insights using Amazon Transcribe and a LLM \n
            - **[GenAI ChatAway](/GenAI_ChatAway):** Conversational chat-style interaction using a LLM \n
            - **[GenAI Content Analyzer](/GenAI_content_analyzer):** Upload images, MS Word, Excel, PPT, PDF, CSV, HTML, or text files, get a summary, auto-prompts and conversational insights using Amazom Rekognition and a LLM \n
            - **[GenAI Product Recommender](/GenAI_product_recommender):** Personslized recommendations and conversational insights using Amazon Personalize and a LLM \n
            - **[GenAI Image Analyzer](/GenAI_image_analyzer):** Analysis of images, label detection and summarization \n
            - **[GenAI Video Analyzer](/GenAI_video_analyzer):** Analysis of videos, label detection and summarization \n
        ''')
    elif industry == 'Legal':
        st.markdown('''
            ### Available demos \n\n
            - **[GenAI Call Analyzer](/GenAI_call_analyzer):** Call transcription, analytics, summarization, auto-prompts and conversational insights using Amazon Transcribe and a LLM \n
            - **[GenAI ChatAway](/GenAI_ChatAway):** Conversational chat-style interaction using a LLM \n
            - **[GenAI Content Analyzer](/GenAI_content_analyzer):** Upload images, MS Word, Excel, PPT, PDF, CSV, HTML,  or text files, get a summary, auto-prompts and conversational insights using Amazom Rekognition and a LLM \n
            - **[GenAI Document Processor](/GenAI_document_processor):** Summarization of multi-page PDFs, auto-prompts and conversational insights using Amazon Textract and a LLM \n
            - **[GenAI Enterprise Search](/GenAI_enterprise_search_interpreter):** Intelligent enterprise search, auto-prompts and conversational interpretation using Amazon Kendra and a LLM \n
            - **[GenAI Image Analyzer](/GenAI_image_analyzer):** Analysis of images, label detection and summarization \n
            - **[GenAI Video Analyzer](/GenAI_video_analyzer):** Analysis of videos, label detection and summarization \n
        ''')
    elif industry == 'Life Sciences':
        st.markdown('''
            ### Available demos \n\n
            - **[GenAI Call Analyzer](/GenAI_call_analyzer):** Call transcription, analytics, summarization, auto-prompts and conversational insights using Amazon Transcribe and a LLM \n
            - **[GenAI ChatAway](/GenAI_ChatAway):** Conversational chat-style interaction using a LLM \n
            - **[GenAI Content Analyzer](/GenAI_content_analyzer):** Upload images, MS Word, Excel, PPT, PDF, CSV, HTML,  or text files, get a summary, auto-prompts and conversational insights using Amazom Rekognition and a LLM \n
            - **[GenAI Digital Persona](/GenAI_digital_persona):** Talk to a 3D Avatar playing the role of a Pharmocologist
            - **[GenAI Pharma Research](/GenAI_pharma_research):** Drug analysis/research using auto-prompts with conversational insights using Amazon Kendra and a LLM \n
            - **[GenAI Image Analyzer](/GenAI_image_analyzer):** Analysis of images, label detection and summarization \n
            - **[GenAI Video Analyzer](/GenAI_video_analyzer):** Analysis of videos, label detection and summarization \n
        ''')
    if industry == 'Manufacturing':
        st.markdown('''
            ### Available demos \n\n
            - **[GenAI Call Analyzer](/GenAI_call_analyzer):** Call transcription, analytics, summarization, auto-prompts and conversational insights using Amazon Transcribe and a LLM \n
            - **[GenAI ChatAway](/GenAI_ChatAway):** Conversational chat-style interaction using a LLM \n
            - **[GenAI Content Analyzer](/GenAI_content_analyzer):** Upload images, csv or text files, get a summary, auto-prompts and conversational insights using Amazom Rekognition and a LLM \n
            - **[GenAI Image Analyzer](/GenAI_image_analyzer):** Analysis of images, label detection and summarization \n
            - **[GenAI Video Analyzer](/GenAI_video_analyzer):** Analysis of videos, label detection and summarization \n
        ''')
    elif industry == 'Fashion':
        st.markdown('''
            ### Available demos \n\n
            - **[GenAI Call Analyzer](/GenAI_call_analyzer):** Call transcription, analytics, summarization, auto-prompts and conversational insights using Amazon Transcribe and a LLM \n
            - **[GenAI ChatAway](/GenAI_ChatAway):** Conversational chat-style interaction using a LLM \n
            - **[GenAI Content Analyzer](/GenAI_content_analyzer):** Upload images, MS Word, Excel, PPT, PDF, CSV, HTML,  or text files, get a summary, auto-prompts and conversational insights using Amazom Rekognition and a LLM \n
            - **[GenAI Product Ideator](/GenAI_product_ideator):** Create product images, description and a press release with just a few words \n
            - **[GenAI Image Analyzer](/GenAI_image_analyzer):** Analysis of images, label detection and summarization \n
            - **[GenAI Video Analyzer](/GenAI_video_analyzer):** Analysis of videos, label detection and summarization \n
        ''')
    elif industry == 'Retail':
        st.markdown('''
            ### Available demos \n\n
            - **[GenAI Call Analyzer](/GenAI_call_analyzer):** Call transcription, analytics, summarization, auto-prompts and conversational insights using Amazon Transcribe and a LLM \n
            - **[GenAI ChatAway](/GenAI_ChatAway):** Conversational chat-style interaction using a LLM \n
            - **[GenAI Content Analyzer](/GenAI_content_analyzer):** Upload images, MS Word, Excel, PPT, PDF, CSV, HTML, or text files, get a summary, auto-prompts and conversational insights using Amazom Rekognition and a LLM \n
            - **[GenAI Product Ideator](/GenAI_product_ideator):** Create product images, description and a press release with just a few words \n
            - **[GenAI Image Analyzer](/GenAI_image_analyzer):** Analysis of images, label detection and summarization \n
            - **[GenAI Video Analyzer](/GenAI_video_analyzer):** Analysis of videos, label detection and summarization \n
        ''')
    elif industry == 'Transport':
        st.markdown('''
            ### Available demos \n\n
            - **[GenAI Call Analyzer](/GenAI_call_analyzer):** Call transcription, analytics, summarization, auto-prompts and conversational insights using Amazon Transcribe and a LLM \n
            - **[GenAI ChatAway](/GenAI_ChatAway):** Conversational chat-style interaction using a LLM \n
            - **[GenAI Content Analyzer](/GenAI_content_analyzer):** Upload images, MS Word, Excel, PPT, PDF, CSV, HTML,  or text files, get a summary, auto-prompts and conversational insights using Amazom Rekognition and a LLM \n
            - **[GenAI Product Ideator](/GenAI_product_ideator):** Create product images, description and a press release with just a few words \n
            - **[GenAI Image Analyzer](/GenAI_image_analyzer):** Analysis of images, label detection and summarization \n
            - **[GenAI Video Analyzer](/GenAI_video_analyzer):** Analysis of videos, label detection and summarization \n
        ''')
    elif industry == 'Technology':
        st.markdown('''
            ### Available demos \n\n
            - **[GenAI Call Analyzer](/GenAI_call_analyzer):** Call transcription, analytics, summarization, auto-prompts and conversational insights using Amazon Transcribe and a LLM \n
            - **[GenAI ChatAway](/GenAI_ChatAway):** Conversational chat-style interaction using a LLM \n
            - **[GenAI Content Analyzer](/GenAI_content_analyzer):** Upload images, MS Word, Excel, PPT, PDF, CSV, HTML, or text files, get a summary, auto-prompts and conversational insights using Amazom Rekognition and a LLM \n
            - **[GenAI Product Ideator](/GenAI_product_ideator):** Create product images, description and a press release with just a few words \n
            - **[GenAI Image Analyzer](/GenAI_image_analyzer):** Analysis of images, label detection and summarization \n
            - **[GenAI Video Analyzer](/GenAI_video_analyzer):** Analysis of videos, label detection and summarization \n
        ''')
with tab2:
    usecase = st.selectbox(
        '**Select an usecase to begin**',
        ('Select...', 'Customer Experience', 'Productivity Improvement', 'Document Processing', 'Enterprise Search', 'Content Generation', 'Computer Vision', 'Product Recommendation', 'Geolocation'))

    if usecase == 'Customer Experience':
        st.markdown('''
            ### Available demos \n\n
            - **[GenAI Call Analyzer](/GenAI_call_analyzer):** Call transcription, analytics, summarization, auto-prompts and conversational insights using Amazon Transcribe and a LLM \n
            - **[GenAI ChatAway](/GenAI_ChatAway):** Conversational chat-style interaction using a LLM \n
            - **[GenAI Digital Persona](/GenAI_digital_persona):** Digital CX specialists
        ''')
    elif usecase == 'Productivity Improvement':
        st.markdown('''
            ### Available demos
            - **[GenAI Agile Guru](/GenAI_Agile_Guru):** Generate Agile Sprint artifacts in seconds \n
            - **[GenAI APPlause](/GenAI_APPlause):** Generate full stack code for apps in seconds along with code quality and performance improvement suggestions \n
            - **[GenAI ChatAway](/GenAI_ChatAway):** Conversational chat-style interaction using a LLM \n
            - **[GenAI Content Analyzer](/GenAI_content_analyzer):** Upload images, csv or text files, get a summary, auto-prompts and conversational insights using Amazom Rekognition and a LLM \n
            - **[GenAI Code Migrator](/GenAI_code_migrator):** Convert source code across programming languages \n
            - **[GenAI Cool Cucumber](/GenAI_cool_cucumber):** Generation of Behavior Driven Development scripts in Cucumber from user stories using an LLM \n
            - **[GenAI Energy Upstream Analyzer](/GenAI_energy_upstream_analyzer):** Well log interpretation of IoT data, analysis of wellsite images, auto-prompts and conversational insights using a LLM \n
            - **[GenAI Graph Guru](/GenAI_Graph_Guru):** Fine-grained insights from graph data analyzed using an LLM \n
            - **[GenAI Pharma Research](/GenAI_pharma_research):** Drug analysis/research using auto-prompts with conversational insights using Amazon Kendra and a LLM \n
        ''')   
    elif usecase == 'Document Processing':
        st.markdown('''
            ### Available demos \n\n
            - **[GenAI Content Analyzer](/GenAI_content_analyzer):** Upload images, csv or text files, get a summary, auto-prompts and conversational insights using Amazom Rekognition and a LLM \n
            - **[GenAI Document Processor](/GenAI_document_processor):** Summarization of multi-page PDFs, auto-prompts and conversational insights using Amazon Textract and a LLM \n
        ''')
    elif usecase == 'Enterprise Search':
        st.markdown('''
            ### Available demos \n\n
            - **[GenAI Enterprise Search](/GenAI_enterprise_search_interpreter):** Intelligent enterprise search, auto-prompts and conversational interpretation using Amazon Kendra and a LLM \n
            - **[GenAI Pharma Research](/GenAI_pharma_research):** Drug analysis/research using auto-prompts with conversational insights using Amazon Kendra and a LLM \n
        ''')
    elif usecase == 'Content Generation':
        st.markdown('''
            ### Available demos \n\n
            - **[GenAI Agile Guru](/GenAI_Agile_Guru):** Generate Agile Sprint artifacts in seconds \n
            - **[GenAI APPlause](/GenAI_APPlause):** Generate full stack code for apps in seconds along with code quality and performance improvement suggestions \n
            - **[GenAI ChatAway](/GenAI_ChatAway):** Conversational chat-style content generation using a LLM \n
            - **[GenAI Demo Reference Architectures](/GenAI_Demo_Reference_Architectures):** Reference architectures of these demos with generated abstract and explanation \n
            - **[GenAI Product Ideator](/GenAI_product_ideator):** Create product images, description and a press release with just a few words \n
        ''')
    elif usecase == 'Computer Vision':
        st.markdown('''
            ### Available demos \n\n
            - **[GenAI Image Analyzer](/GenAI_image_analyzer):** Analysis of images, label detection and summarization \n
            - **[GenAI Video Analyzer](/GenAI_video_analyzer):** Analysis of videos, label detection and summarization \n
        ''')
    elif usecase == 'Product Recommendation':
        st.markdown('''
            ### Available demos \n\n
            - **[GenAI Product Recommender](/GenAI_product_recommender):** Personslized recommendations and conversational insights using Amazon Personalize and a LLM \n
            
        ''')
    elif usecase == 'Geolocation':
        st.markdown('''
            ### Available demos \n\n
            - **[GenAI Geospatial Analyzer](/GenAI_geospatial_analyzer):** Search, geocoding and analysis of physical locations in a map using Amazon Location Service and a LLM \n
        ''')    
with tab3:
    function = st.selectbox(
        '**Select a job function to begin**',
        ('Select...', 'Research & Development', 'Product Management', 'Sales & Marketing', 'Engineering', 'Content Analysis', 'Management'))

    if function == 'Research & Development':
        st.markdown('''
            ### Available demos \n\n
            - **[GenAI Agile Guru](/GenAI_Agile_Guru):** Generate Agile Sprint artifacts in seconds \n
            - **[GenAI APPlause](/GenAI_APPlause):** Generate full stack code for apps in seconds along with code quality and performance improvement suggestions \n
            - **[GenAI ChatAway](/GenAI_ChatAway):** Conversational chat-style knowledge curation using a LLM \n
            - **[GenAI Content Analyzer](/GenAI_content_analyzer):** Upload images, csv or text files, get a summary, auto-prompts and conversational insights using an LLM \n
            - **[GenAI Document Processor](/GenAI_document_processor):** Summarization of multi-page PDFs and conversational insights using Amazon Textract and a LLM \n
            - **[GenAI Digital Persona](/GenAI_digital_persona):** Talk to a 3D Avatar playing the role of an Automotive Specialist
            - **[GenAI Enterprise Search](/GenAI_enterprise_search_interpreter):** Intelligent enterprise search, auto-prompts and conversational interpretation using Amazon Kendra and a LLM \n
            - **[GenAI Image Analyzer](/GenAI_image_analyzer):** Analysis of images, label detection and summarization \n
            - **[GenAI Video Analyzer](/GenAI_video_analyzer):** Analysis of videos, label detection and summarization \n
            - **[GenAI Energy Upstream Analyzer](/GenAI_energy_upstream_analyzer):** Well log interpretation of IoT data, analysis of wellsite images, auto-prompts and conversational insights using a LLM \n
            - **[GenAI Graph Guru](/GenAI_Graph_Guru):** Fine-grained insights from graph data analyzed using an LLM \n
            - **[GenAI Pharma Research](/GenAI_pharma_research):** Drug analysis/research using auto-prompts with conversational insights using Amazon Kendra and a LLM \n
        ''')
    elif function == 'Product Management':
        st.markdown('''
            ### Available demos
            - **[GenAI ChatAway](/GenAI_ChatAway):** Conversational chat-style interaction using a LLM \n
            - **[GenAI Product Ideator](/GenAI_product_ideator):** Create product images, description and a press release with just a few words \n
        ''')   
    elif function == 'Sales & Marketing':
        st.markdown('''
            ### Available demos \n\n
            - **[GenAI ChatAway](/GenAI_ChatAway):** Conversational chat-style interaction using a LLM \n
            - **[GenAI Digital Persona](/GenAI_digital_persona):** Talk to a 3D Avatar playing the role of a Financial Analyst
            - **[GenAI Enterprise Search](/GenAI_enterprise_search_interpreter):** Intelligent enterprise search, auto-prompts and conversational interpretation using Amazon Kendra and a LLM \n
            - **[GenAI Product Ideator](/GenAI_product_ideator):** Create product images, description and a press release with just a few words \n
        ''')
    elif function == 'Engineering':
        st.markdown('''
            ### Available demos \n\n
            - **[GenAI Agile Guru](/GenAI_Agile_Guru):** Generate Agile Sprint artifacts in seconds \n
            - **[GenAI APPlause](/GenAI_APPlause):** Generate full stack code for apps in seconds along with code quality and performance improvement suggestions \n
            - **[GenAI ChatAway](/GenAI_ChatAway):** Conversational chat-style interaction using a LLM \n
            - **[GenAI Content Analyzer](/GenAI_content_analyzer):** Upload images, MS Word, Excel, PPT, PDF, CSV, HTML, or text files, get a summary, auto-prompts and conversational insights using Amazom Rekognition and a LLM \n
            - **[GenAI Cool Cucumber](/GenAI_cool_cucumber):** Generation of Behavior Driven Development scripts in Cucumber from user stories using an LLM \n
            - **[GenAI Image Analyzer](/GenAI_image_analyzer):** Analysis of images, label detection and summarization \n
            - **[GenAI Video Analyzer](/GenAI_video_analyzer):** Analysis of videos, label detection and summarization \n
        ''')
    elif function == 'Content Analysis':
        st.markdown('''
            ### Available demos \n\n
            - **[GenAI Call Analyzer](/GenAI_call_analyzer):** Call transcription, analytics, summarization, auto-prompts and conversational insights using Amazon Transcribe and a LLM \n
            - **[GenAI Content Analyzer](/GenAI_content_analyzer):** Upload images, MS Word, Excel, PPT, PDF, CSV, HTML, or text files, get a summary, auto-prompts and conversational insights using Amazom Rekognition and a LLM \n
            - **[GenAI Graph Guru](/GenAI_Graph_Guru):** Fine-grained insights from graph data analyzed using an LLM \n
            - **[GenAI Image Analyzer](/GenAI_image_analyzer):** Analysis of images, label detection and summarization \n
            - **[GenAI Video Analyzer](/GenAI_video_analyzer):** Analysis of videos, label detection and summarization \n
        ''')
    elif function == 'Management':
        st.markdown('''
            ### Available demos \n\n
            - **[GenAI ChatAway](/GenAI_ChatAway):** Conversational chat-style interaction using a LLM \n
            - **[GenAI Content Analyzer](/GenAI_content_analyzer):** Upload images, MS Word, Excel, PPT, PDF, CSV, HTML,  or text files, get a summary, auto-prompts and conversational insights using Amazom Rekognition and a LLM \n
            - **[GenAI Enterprise Search](/GenAI_enterprise_search_interpreter):** Intelligent enterprise search, auto-prompts and conversational interpretation using Amazon Kendra and a LLM \n
            - **[GenAI Image Analyzer](/GenAI_image_analyzer):** Analysis of images, label detection and summarization \n
        ''')

st.markdown(
    """
    Learn the Generative AI art-of-the-possible with these awesome demos for popular use cases!
    Using a combination of purpose built AWS AI services with powerful LLMs you can go from ideation to implementation in **a few minutes!!**
    Be ready to be wowed!!
    
    - [How did we make these awesome demos](/GenAI_Demo_Reference_Architectures)

    ### Learn more about AWS AI services
    
    - [Pre-trained AI services](https://aws.amazon.com/machine-learning/ai-services/)
    - [AI usecases explorer](https://aiexplorer.aws.amazon.com/?lang=en&trk=47702943-c5e6-44e8-841f-d061a5468505&sc_channel=el)
   
    ### Learn more about AWS Generative AI announcements

    - [Amazon Bedrock](https://aws.amazon.com/bedrock/)
    - [Amazon CodeWhisperer](https://aws.amazon.com/codewhisperer/)
    - AWS Accelerated Computing [training](https://aws.amazon.com/blogs/aws/amazon-ec2-trn1-instances-for-high-performance-model-training-are-now-available/) and [inference for Generative AI](https://aws.amazon.com/blogs/aws/amazon-ec2-inf2-instances-for-low-cost-high-performance-generative-ai-inference-are-now-generally-available)


    """
)


import streamlit as st
import streamlit.components.v1 as components

# embed streamlit docs in a streamlit app


st.set_page_config(page_title="GenAI Digital Persona", page_icon="thought_balloon")

st.markdown("# Talk to Digital Humans powered by Generative Conversational AI")
st.markdown("New: Digital humans can now converse in multiple languages")
st.sidebar.header("GenAI Digital Persona")

values = ['Financial Analyst', 'Geo-Scientist', 'Pharmacologist', 'Automotive Specialist']
default_ix = values.index('Financial Analyst')
persona = st.selectbox('Select a persona', values, index=default_ix)

if persona.lower() == 'financial analyst':
	st.sidebar.markdown("""
		### Digital Financial Analyst's knowledge source \n\n
		- Bank of America 10K docs and annual reports
		- Factset annual report
		- Moody's annual report
		- Morningstar's annual report
		- S & P annual report
	""")
	st.sidebar.markdown("""
        ### Example questions you can ask \n\n
        What is Earnings per share and what does it mean? \n
        Qu'est-ce que la comptabilité d'exercice ?
        Was sind die Vor- und Nachteile einer Aktienvergütung? \n
        Why are EU members not aligned on fiscal policy? \n
        What was Morningstar's annual revenue in 2022? \n
    """)

	st.markdown("""
    	<a href="https://ddna-amazon-web-services--genai-financial-ana.soului.dh.soulmachines.cloud/?sig=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE2ODMxMDI3NDcsImlzcyI6InNpZ25lZF91cmwtZjdkYTBhZGMtZjUxOC00Y2Y3LTliMWEtNjAxMjY4YjRiNjcxIiwiZXhwIjoxNzY5NDE2MzQ3LCJlbWFpbCI6ImFtYXpvbi13ZWItc2VydmljZXMtLWdlbmFpLWZpbmFuY2lhbC1hbmFAZGRuYS5zdHVkaW8iLCJzb3VsSWQiOiJkZG5hLWFtYXpvbi13ZWItc2VydmljZXMtLWdlbmFpLWZpbmFuY2lhbC1hbmEifQ.TzCQ-MLP_r1X76eMRxDTV4L54YtOdxlhgYYiz2QW65U" target = "_blank"> 
        	Talk to Julien - Financial Analyst
    	</a>
	""", unsafe_allow_html=True)

if persona.lower() == 'geo-scientist':
	st.sidebar.markdown("""
        ### Example questions you can ask \n\n
        What is well log interpretation? \n
        What happens in a refinery? \n
        What is OSDU? \n
    """)

	st.markdown("""
    	<a href="https://ddna-amazon-web-services--genai-automotive-sp.soului.dh.soulmachines.cloud/?sig=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE2ODMzNzM0NzUsImlzcyI6InNpZ25lZF91cmwtYzM4YzM3YTktN2ZmNi00YjVmLThkYjktM2NhOTdlMWEzZWVhIiwiZXhwIjoxNzY5Njg3MDc1LCJlbWFpbCI6ImFtYXpvbi13ZWItc2VydmljZXMtLWdlbmFpLWF1dG9tb3RpdmUtc3BAZGRuYS5zdHVkaW8iLCJzb3VsSWQiOiJkZG5hLWFtYXpvbi13ZWItc2VydmljZXMtLWdlbmFpLWF1dG9tb3RpdmUtc3AifQ.WH20RFM6DGGlSYdjqn2zVOiHBgzchUD7nGYfrCYBYiQ" target = "_blank"> 
        	Talk to Camile - Geo-Scientist
    	</a>
	""", unsafe_allow_html=True)
if persona.lower() == 'pharmacologist':
	st.sidebar.markdown("""
        ### Example questions you can ask \n\n
        What is aspirin? \n
        Are there contra indications between Aspirin and Paracetomol? \n
        Are there any treatment options for novel coronavirus? \n
    """)

	st.markdown("""
    	<a href="https://ddna-amazon-web-services--genai-pharmacologis.soului.dh.soulmachines.cloud/?sig=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE2ODMzNzIzODYsImlzcyI6InNpZ25lZF91cmwtMDZhNDFlZjQtOWE3YS00ZTk4LWI1ZjgtNTVlODAwZWFhMTc1IiwiZXhwIjoxNzY5Njg1OTg2LCJlbWFpbCI6ImFtYXpvbi13ZWItc2VydmljZXMtLWdlbmFpLXBoYXJtYWNvbG9naXNAZGRuYS5zdHVkaW8iLCJzb3VsSWQiOiJkZG5hLWFtYXpvbi13ZWItc2VydmljZXMtLWdlbmFpLXBoYXJtYWNvbG9naXMifQ.RAIThB_g-YEB5K1pn7bMA89RLTr1buAao8mVve7IoLI" target = "_blank"> 
        	Talk to Anele - Pharmacologist
    	</a>
	""", unsafe_allow_html=True)
if persona.lower() == 'automotive specialist':
	st.sidebar.markdown("""
        ### Example questions you can ask \n\n
        What are the steps for an oil change on a BMW 3 series? \n
        How can I reset a tire pressure indicator in a Toyota Camry? \n
    """)

	st.markdown("""
    	<a href="https://ddna-amazon-web-services--genai-automotive-sp-3e6fd3b.soului.dh.soulmachines.cloud/?sig=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE2ODMzNzMwNzUsImlzcyI6InNpZ25lZF91cmwtZjFiNzQ3YjktNGY2ZC00YTQ0LWFmOTUtMDk0YzRjNDczNmZlIiwiZXhwIjoxNzY5Njg2Njc1LCJlbWFpbCI6ImFtYXpvbi13ZWItc2VydmljZXMtLWdlbmFpLWF1dG9tb3RpdmUtc3AtM2U2ZmQzYkBkZG5hLnN0dWRpbyIsInNvdWxJZCI6ImRkbmEtYW1hem9uLXdlYi1zZXJ2aWNlcy0tZ2VuYWktYXV0b21vdGl2ZS1zcC0zZTZmZDNiIn0.z3QvYtEPX9H9dJpR_casybFBbuQ57XbCyRo9pq8mcX4" target = "_blank"> 
        	Talk to Amari - Automotive Specialist
    	</a>
	""", unsafe_allow_html=True)

#components.iframe("https://ddna-amazon-web-services--digital-executive-s.soului.dh.soulmachines.cloud/session")
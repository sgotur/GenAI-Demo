import asyncio
from aiohttp import ClientSession
import ai21
import streamlit as st


def complete(model_type, prompt, config):
    return ai21.Completion.execute(sm_endpoint="j2-jumbo-instruct", prompt=prompt, **config)

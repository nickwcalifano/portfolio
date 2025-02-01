""" This module has functions to call Ollama and Azure OpenAI endpoints """
import time
from typing import Literal, Union
import os
import json
import aiohttp
import numpy as np

# Define GPT endpoints and retry parameters
URL_GPT4O_MINI_COMPLETE = "https://{endpoint}/openai/deployments/{deployment-id}/completions?api-version=2024-10-21"
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
NUM_REQUEST_RETRIES = 10 # If we have hit our token quota, we wait some time then try again (exponential backoff)
MAX_WAIT_TIME = 60 # Max time in seconds to sleep each time we wait
GPT_4O_HEADERS = {
    "Content-Type": "application/json",
    "api-key": AZURE_OPENAI_KEY
}

# Define ollama endpoints
URL_OLLAMA_GEN = "http://localhost:11434/api/generate"
ALLOWED_MODEL_STRINGS = Literal["deepseek-r1:14b", "llama3.2-vision", "phi4", "gpt4o-mini", "gpt4o"]

def get_model_type(model_name: ALLOWED_MODEL_STRINGS) -> str:
    """ Returns llama if the model is a llama model and returns gpt if the model is a gpt model
    """
    if "gpt" in model_name:
        return "azure_openai"
    return "ollama"


async def call_cached_llm(
        model_name: ALLOWED_MODEL_STRINGS,
        session: aiohttp.ClientSession,
        request_payload: dict,
        cache: dict,
        returned_stopped_only: bool=True
    ) -> Union[str, None]:
    """ Wrapper around call_llm. Uses cached result if available, and saves result to cache if it wasn't available
    """
    # Get the model type (ollama or GPT)
    model_type = get_model_type(model_name)

    # Get a hashable version of the payload - we will be caching the results in a dict
    if model_type == "ollama":
        hashable_payload = tuple(
            [model_name, request_payload["prompt"]] +
            [tuple(option) for option in request_payload["options"].items()]
        )
    else:
        hashable_payload = tuple(
            [model_name, request_payload["prompt"]] +
            [request_payload[item] for item in ["temperature", "top_p"]]
        )

    # If this request has already been sent, use the cached version
    if hashable_payload in cache:
        return cache[hashable_payload]

    # Since the request is new, make a call to the LLM and save the result in the cache
    result = await call_llm(model_name, session, request_payload, returned_stopped_only)
    cache[hashable_payload] = result
    return result


async def call_llm(
        model_name: ALLOWED_MODEL_STRINGS,
        session: aiohttp.ClientSession,
        request_payload: dict,
        returned_stopped_only: bool=True
    ) -> Union[str, None]:
    """ Calls the given model's generate endpoint using the given request payload
    """
    # Get the model type (ollama or GPT)
    model_type = get_model_type(model_name)

    # Llama and Azure OpenAI have slightly different APIs, so we need to call them slightly differently
    if model_type == "ollama":
        return await call_ollama(session, request_payload, returned_stopped_only)
    return await call_azure_openai(session, request_payload, returned_stopped_only)


async def call_ollama(
    session: aiohttp.ClientSession,
    request_payload: dict,
    returned_stopped_only: bool=True
):
    """ Calls an ollama generate endpoint using the given request payload
    """
    try:
        async with session.post(URL_OLLAMA_GEN, json=request_payload) as resp:
            response = await resp.json()
            resp.raise_for_status()

        # If the model hits the output token limit, simply return None
        if returned_stopped_only and response["done_reason"] != "stop":
            retval = None
        else:
            retval = response["response"]

    except aiohttp.ClientResponseError as e:
        print(
            "LLAMA ClientResponseError", e,
            "\nRequest:", request_payload,
            "\nResponse:", json.dumps(response)
        )
        return None

    except KeyError as e:
        print("KeyError", e)
        raise e

    return retval


async def call_azure_openai(
    session: aiohttp.ClientSession,
    request_payload: dict,
    returned_stopped_only: bool=True
):
    """ Calls an Azure OpenAI generate endpoint using the given request payload
    """
    for i in range(NUM_REQUEST_RETRIES):
        try:
            async with session.post(URL_GPT4O_MINI_COMPLETE, headers=GPT_4O_HEADERS, json=request_payload) as resp:
                response = await resp.json()
                resp.raise_for_status(session, request_payload, returned_stopped_only)

            # If the model hits the output token limit, simply return None
            if returned_stopped_only and response["choices"][0]["finish_reason"] != "stop":
                retval = None
            else:
                retval = response["choices"][0]["text"]

            # If the model returns a valid response, there's no need for further retries. Break out of the for-loop
            break

        except aiohttp.ClientResponseError as e:
            # 400 is for content filter. If the request failed due to content filtering, we want to add it to the
            #   cache because retrying the same prompt will be futile
            if resp.status == 400:
                retval = None
                break

            # 429 is for quote rate limit reached. If the request failed due to quota, wait then try again
            # Similarly, if it is a 500 error, wait then try again to see if it fixes itself
            if resp.status == 429 or 500 <= resp.status < 600:
                time.sleep(min(MAX_WAIT_TIME, np.power(2, i)))

            # If the request failed due to any other reason, print the error, don't save the result to cache, and
            #   return None. But don't raise an error (so the processing continues)
            # If this was production code rather than an analysis of a novel technique, I would have better error
            #   handling. Since we are iterating over the hyperparameter search space (number of questions), running
            #   the notebook takes a long time, and I don't want a transient network error to kill the process
            # Rerunning the notebook will retry for these errors
            else:
                print(
                    "GPT ClientResponseError", e,
                    "\nRequest:", request_payload,
                    "\nResponse:", json.dumps(response)
                )
                return None
    return retval

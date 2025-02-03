""" This module has functions to call Ollama and Azure OpenAI endpoints """
import asyncio
import json
import os
from typing import Literal, Union

import aiohttp
import numpy as np

# Constants for retry logic
NUM_REQUEST_RETRIES = 10 # If we have hit our token quota, we wait some time then try again (exponential backoff)
MAX_WAIT_TIME = 60 # Max time in seconds to sleep each time we wait

# Type aliases
AllowedModelNames  = Literal["deepseek-r1:14b", "llama3.2-vision", "phi4", "gpt4o-mini", "gpt4o"]
ServerTypes = Literal["azure_openai", "ollama"]

class AzureOpenAiRetriesExhaustedError(Exception):
    """ Azure OpenAI Service retries have been exhausted."""

class LlmUtils():
    """ This class has utility functions to call Ollama and Azure OpenAI endpoints """

    # Configuration from environment variables (checked at startup)
    AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
    AZURE_OPENAI_URL = os.environ.get("AZURE_OPENAI_URL")
    OLLAMA_URL = os.environ.get("OLLAMA_URL")

    if not all([AZURE_OPENAI_KEY, AZURE_OPENAI_URL, OLLAMA_URL]):
        print(AZURE_OPENAI_KEY, AZURE_OPENAI_URL, OLLAMA_URL)
        missing_vars = [var for var, val in
                        {"AZURE_OPENAI_KEY": AZURE_OPENAI_KEY,
                         "AZURE_OPENAI_URL": AZURE_OPENAI_URL,
                         "OLLAMA_URL": OLLAMA_URL}.items() if val is None]
        raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

    @staticmethod
    def get_model_type(model_name: AllowedModelNames) -> ServerTypes:
        """Returns the server type based on the model name."""
        return "azure_openai" if "gpt" in model_name else  "ollama"

    @staticmethod
    def create_hashable_payload(model_name, request_payload):
        """Creates a hashable version of the request payload for caching."""
        model_type = LlmUtils.get_model_type(model_name)
        if model_type == "ollama":
            return (
                model_name,
                request_payload["prompt"],
                tuple(tuple(option) for option in request_payload["options"].items())
            )

        # Azure OpenAI
        return (
            model_name,
            request_payload["messages"][0]["content"], request_payload["messages"][1]["content"],
            tuple(request_payload[item] for item in ["temperature", "top_p"])
        )

    @classmethod
    async def call_cached_llm(
            cls,
            model_name: AllowedModelNames,
            session: aiohttp.ClientSession,
            request_payload: dict,
            cache: dict,
            returned_stopped_only: bool=True
        ) -> Union[str, None]:
        """Wrapper around call_llm. Uses cached result if available. Saves new result to cache."""
        hashable_payload = cls.create_hashable_payload(model_name, request_payload)

        if hashable_payload in cache:
            return cache[hashable_payload]

        result = await cls.call_llm(model_name, session, request_payload, returned_stopped_only)
        if result is not None:
            cache[hashable_payload] = result
        return result

    @classmethod
    async def call_llm(
            cls,
            model_name: AllowedModelNames,
            session: aiohttp.ClientSession,
            request_payload: dict,
            returned_stopped_only: bool=True
        ) -> Union[str, None]:
        """Calls the given model's generate endpoint."""
        model_type = cls.get_model_type(model_name)
        handler = cls.call_ollama if model_type == "ollama" else cls.call_azure_openai
        return await handler(session, request_payload, returned_stopped_only)

    @classmethod
    async def call_ollama(
        cls,
        session: aiohttp.ClientSession,
        request_payload: dict,
        returned_stopped_only: bool=True
    ) -> Union[str, None]:
        """Calls an Ollama generate endpoint."""
        try:
            async with session.post(cls.OLLAMA_URL, json=request_payload) as resp:
                response = await resp.json()
                resp.raise_for_status()

            # If the model hits the output token limit, simply return None
            if returned_stopped_only and response["done_reason"] != "stop":
                return None
            return response["response"]

        except aiohttp.ClientResponseError as e:
            print(f"LLAMA ClientResponseError {e}\nRequest: {request_payload}\nResponse: {json.dumps(response)}")
            return None
        except Exception as e:
            print(f"Ollama Error: {e}\nRequest: {request_payload}")
            return None

    @classmethod
    async def call_azure_openai(
        cls,
        session: aiohttp.ClientSession,
        request_payload: dict,
        returned_stopped_only: bool=True
    ) -> Union[str, None]:
        """Calls an Azure OpenAI generate endpoint."""
        headers = {
            "Content-Type": "application/json",
            "api-key": cls.AZURE_OPENAI_KEY,
        }
        for i in range(NUM_REQUEST_RETRIES):
            try:
                async with session.post(cls.AZURE_OPENAI_URL, headers=headers, json=request_payload) as resp:
                    response = await resp.json()
                    resp.raise_for_status()

                # If the model hits the output token limit, simply return None
                if returned_stopped_only and response["choices"][0]["finish_reason"] != "stop":
                    return None
                return response["choices"][0]["message"]["content"]

            except aiohttp.ClientResponseError as e:
                # 400 is for ill-formatted request payload as well as for content filter flags
                if resp.status == 400:
                    print(f"GPT ClientResponseError {e}\nRequest: {request_payload}\nResponse: {json.dumps(response)}")
                    return None

                # 429 is for quote rate limit reached. If the request failed due to quota, wait then try again
                # Similarly, if it is a 500 error, wait then try again to see if it fixes itself
                if resp.status == 429 or 500 <= resp.status < 600:
                    await asyncio.sleep(min(MAX_WAIT_TIME, np.power(2, i)))

                # If the request failed due to any other reason: print the request and response then raise an error
                # If this was production code, I would have better error handling
                else:
                    print(f"GPT ClientResponseError {e}\nRequest: {request_payload}\nResponse: {json.dumps(response)}")
                    raise e

        raise AzureOpenAiRetriesExhaustedError("Max retries exceeded for Azure OpenAI")

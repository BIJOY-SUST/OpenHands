import copy
import os
import time
import warnings
from functools import partial
from typing import Any

import requests
from openhands.core.utils import json

from openhands.core.config import LLMConfig

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import litellm  # Commented out since we're not using it directly now


from litellm import Choices, Message, OpenAI
from litellm.types.utils import CostPerToken, ModelResponse, Usage   
from litellm import Message as LiteLLMMessage
from litellm import ModelInfo, PromptTokensDetails
from litellm import completion as litellm_completion
from litellm import completion_cost as litellm_completion_cost
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
)
from litellm.types.utils import CostPerToken, ModelResponse, Usage

from openhands.core.exceptions import CloudFlareBlockageError
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message
from openhands.llm.debug_mixin import DebugMixin
from openhands.llm.fn_call_converter import (
    STOP_WORDS,
    convert_fncall_messages_to_non_fncall_messages,
    convert_non_fncall_messages_to_fncall_messages,
)
from openhands.llm.metrics import Metrics
from openhands.llm.retry_mixin import RetryMixin

__all__ = ['LLM']

# tuple of exceptions to retry on
LLM_RETRY_EXCEPTIONS: tuple[type[Exception], ...] = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.HTTPError,
)

FUNCTION_CALLING_SUPPORTED_MODELS = [
    # Add your supported models here if needed
]


class LLM(RetryMixin, DebugMixin):
    """The LLM class represents a Language Model instance.

    Attributes:
        config: an LLMConfig object specifying the configuration of the LLM.
    """

    def __init__(
        self,
        config: LLMConfig,
        metrics: Metrics | None = None,
    ):
        """Initializes the LLM.

        Args:
            config: The LLM configuration.
            metrics: The metrics to use.
        """
        self.metrics: Metrics = (
            metrics if metrics is not None else Metrics(model_name=config.model)
        )
        self.cost_metric_supported: bool = False  # Set to False since we're not calculating cost
        self.config: LLMConfig = copy.deepcopy(config)

        if self.config.log_completions:
            if self.config.log_completions_folder is None:
                raise RuntimeError(
                    'log_completions_folder is required when log_completions is enabled'
                )
            os.makedirs(self.config.log_completions_folder, exist_ok=True)

        if self.is_function_calling_active():
            logger.debug('LLM: model supports function calling')

        self._completion_unwrapped = self._completion_flask

        @self.retry_decorator(
            num_retries=self.config.num_retries,
            retry_exceptions=LLM_RETRY_EXCEPTIONS,
            retry_min_wait=self.config.retry_min_wait,
            retry_max_wait=self.config.retry_max_wait,
            retry_multiplier=self.config.retry_multiplier,
        )
        def wrapper(*args, **kwargs):
            """Wrapper for the completion function. Logs the input and output."""
            # from openhands.core.utils import json

            messages: list[dict[str, Any]] | dict[str, Any] = []
            mock_function_calling = kwargs.pop('mock_function_calling', False)

            if len(args) > 0:
                messages = args[0]
            elif 'messages' in kwargs:
                messages = kwargs['messages']

            # ensure we work with a list of messages
            messages = messages if isinstance(messages, list) else [messages]
            original_fncall_messages = copy.deepcopy(messages)
            mock_fncall_tools = None
            if mock_function_calling:
                assert (
                    'tools' in kwargs
                ), "'tools' must be in kwargs when mock_function_calling is True"
                messages = convert_fncall_messages_to_non_fncall_messages(
                    messages, kwargs['tools']
                )
                kwargs['messages'] = messages
                kwargs['stop'] = STOP_WORDS
                mock_fncall_tools = kwargs.pop('tools')

            # if we have no messages, something went very wrong
            if not messages:
                raise ValueError(
                    'The messages list is empty. At least one message is required.'
                )

            # log the entire LLM prompt
            self.log_prompt(messages)

            try:
                # Call the Flask API
                resp = self._completion_unwrapped(*args, **kwargs)

                non_fncall_response = copy.deepcopy(resp)
                if mock_function_calling:
                    assert len(resp.choices) == 1
                    assert mock_fncall_tools is not None
                    non_fncall_response_message = resp.choices[0].message
                    fn_call_messages_with_response = (
                        convert_non_fncall_messages_to_fncall_messages(
                            messages + [non_fncall_response_message], mock_fncall_tools
                        )
                    )
                    fn_call_response_message = fn_call_messages_with_response[-1]
                    resp.choices[0].message = fn_call_response_message

                # log for evals or other scripts that need the raw completion
                if self.config.log_completions:
                    assert self.config.log_completions_folder is not None
                    log_file = os.path.join(
                        self.config.log_completions_folder,
                        f'{self.metrics.model_name.replace("/", "__")}-{time.time()}.json',
                    )

                    _d = {
                        'messages': messages,
                        'response': resp,
                        'args': args,
                        'kwargs': {k: v for k, v in kwargs.items() if k != 'messages'},
                        'timestamp': time.time(),
                        'cost': 0.0,  # Cost calculation is not supported here
                    }
                    if mock_function_calling:
                        _d['response'] = non_fncall_response
                        _d['fncall_messages'] = original_fncall_messages
                        _d['fncall_response'] = resp
                    with open(log_file, 'w') as f:
                        f.write(json.dumps(_d))

                message_back: str = resp['choices'][0]['message']['content']
                tool_calls = resp['choices'][0]['message'].get('tool_calls', [])
                if tool_calls:
                    for tool_call in tool_calls:
                        fn_name = tool_call.function.name
                        fn_args = tool_call.function.arguments
                        message_back += f'\nFunction call: {fn_name}({fn_args})'

                # log the LLM response
                self.log_response(message_back)

                return resp
            except Exception as e:
                raise e

        self._completion = wrapper

    @property
    def completion(self):
        """Decorator for the completion function."""
        return self._completion

    def is_function_calling_active(self) -> bool:
        # Adjust as needed
        return self.config.model in FUNCTION_CALLING_SUPPORTED_MODELS

    def _completion_flask(self, *args, **kwargs):
        """Send a request to the Flask API and return the response."""
        # Extract messages
        if len(args) > 0:
            messages = args[0]
        elif 'messages' in kwargs:
            messages = kwargs['messages']
        else:
            raise ValueError('Messages are required for completion.')

        # Ensure messages is a list of dicts
        if isinstance(messages, Message):
            messages = [messages]
        elif isinstance(messages, list):
            messages = messages
        else:
            messages = [messages]

        # Serialize messages    # FIXME: Check later
        # messages = [message.model_dump() for message in messages]

        # Build the payload
        payload = {
            'messages': messages,
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'max_tokens': self.config.max_output_tokens,
            'stop': kwargs.get('stop', None),
            # Include other parameters as needed
        }

        with open(f'payload_{time.time()}.json', 'w') as f:
            f.write(json.dumps(payload))

        logger.debug(f'LLM: Sending request to Flask API')
        # Send the request to the Flask API
        response = requests.post(
            self.config.base_url,  # Flask API endpoint
            headers={
                'Authorization': f'Bearer {self.config.api_key}',
                'Content-Type': 'application/json',
            },
            json=payload,
            timeout=self.config.timeout,
        )
        logger.debug(f'LLM: Received response from Flask API')

        # Check the response status
        if response.status_code != 200:
            raise Exception(
                f'Flask API returned status code {response.status_code}: {response.text}'
            )

        # # Parse the response
        # response_data = response.json()

        # # Build a ModelResponse object similar to what litellm returns
        # # from litellm.types import ModelResponse, Choice, Message as LiteLLMMessage
        # # from litellm.types.utils import CostPerToken, ModelResponse, Usage

        # choices = response_data.get('choices', [])
        # usage = response_data.get('usage', {})
        # logger.debug(f'LLM: Received {len(choices)} choices')
        # logger.debug(f'LLM: Usage: {usage}')
        # # Convert choices to litellm Choice objects
        # choices = [
        #     Choice(
        #         message=LiteLLMMessage(**choice['message']),
        #         finish_reason=choice.get('finish_reason'),
        #         index=choice.get('index', 0),
        #     )
        #     for choice in choices
        # ]

        # # Build the ModelResponse object
        # model_response = ModelResponse(
        #     choices=choices,
        #     usage=usage,
        #     model=response_data.get('model'),
        #     # Add other fields if necessary
        # )

        # Assuming response_data is your JSON response
        response_data = response.json()

        # Log the number of choices and usage
        logger.debug(f'LLM: Received {len(response_data.get("choices", []))} choices')
        logger.debug(f'LLM: Usage: {response_data.get("usage", {})}')

        # Convert choices to litellm Choices objects
        choices = [
            Choices(
                message=LiteLLMMessage(**choice['message']),
                finish_reason=choice.get('finish_reason'),
                index=choice.get('index', 0)
            ) for choice in response_data.get('choices', [])
        ]

        # Create Usage object
        usage = Usage(**response_data.get('usage', {}))

        # Build the ModelResponse object
        model_response = ModelResponse(
            choices=choices,
            usage=usage,
            model=response_data.get('model'),
            created=response_data.get('created')
            # Add other fields as necessary
        )

        return model_response

    def __str__(self):
        if self.config.api_version:
            return f'LLM(model={self.config.model}, api_version={self.config.api_version}, base_url={self.config.base_url})'
        elif self.config.base_url:
            return f'LLM(model={self.config.model}, base_url={self.config.base_url})'
        return f'LLM(model={self.config.model})'

    def __repr__(self):
        return str(self)

    def reset(self) -> None:
        self.metrics.reset()

    def format_messages_for_llm(self, messages: Message | list[Message]) -> list[dict]:
        if isinstance(messages, Message):
            messages = [messages]

        # set flags to know how to serialize the messages
        for message in messages:
            message.cache_enabled = False  # Adjust if needed
            message.vision_enabled = False  # Adjust if needed
            message.function_calling_enabled = self.is_function_calling_active()

        # let pydantic handle the serialization
        return [message.model_dump() for message in messages]

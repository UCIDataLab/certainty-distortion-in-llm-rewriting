from pydantic import BaseModel
from tenacity import RetryCallState, retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any

import copy
import openai
import json, os
import math
import nltk
import torch
import traceback


APIs = ("vllm", "openai")

def _return_error(retry_state):
    import pdb;  pdb.set_trace()
    kwargs = retry_state.args[2] if len(retry_state.args) > 2 else {}
    exception = retry_state.outcome.exception()
    print(
        f"[FAILED after {retry_state.attempt_number} attempts] "
        f"Last exception: {exception}"
    )

    if kwargs.get("return_usage_tokens"):
        return "<ERROR>", kwargs.get("max_tokens", 1000)
    return "<ERROR>"


def print_retry(retry_state: RetryCallState):
    exception = retry_state.outcome.exception()
    wait_time = retry_state.next_action.sleep if retry_state.next_action else None

    print(
        f"[Retry {retry_state.attempt_number}] "
        f"Exception: {exception} | "
        f"Retrying in {wait_time:.2f} seconds..."
    )


class Model:
    def __init__(self, gen_kwargs=None, **kwargs):
        self.gen_kwargs = {} if gen_kwargs is None else gen_kwargs

    def generate(self, messages: List[str], **kwargs):
        raise NotImplementedError("must be override by subclass")
    
    def estimate_tokens(self, text: str, buffer: int=50, **kwargs):
        n_words = len([t.strip() for t in nltk.word_tokenize(text) if t.strip()])
        return int(math.ceil(n_words * 4 / 3)) + buffer


class OpenAIModel(Model):
    def __init__(self, model_name: str, connection_configs=None, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        if not connection_configs:
            connection_configs = json.loads(os.environ.get("CONNECTION_CONFIGS", "{}"))
        self.client = openai.OpenAI(**connection_configs)

    @property
    def client_endpoint(self):
        if "gpt-5-nano" in self.model_name:
            return self.client.responses.create
        else:
            return self.client.chat.completions.create

    def _get_updated_gen_params(self, **kwargs):
        """Updates the generation parameters with the provided kwargs.
        
        If the model is a GPT-5 model, updates the max completion tokens.
        Otherwise, updates the generation parameters normally.
        """
        gen_kwargs = copy.deepcopy(self.gen_kwargs)
        gen_kwargs.update(**kwargs)
        return gen_kwargs

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=print_retry,
        retry_error_callback=_return_error,
        reraise=False,
    )
    def generate(self, messages: List[Dict[str, str]], return_usage_tokens: bool=False, **kwargs) -> str:
        generate_kwargs = self._get_updated_gen_params(**kwargs)
        try:
            chat_response = self.client.chat.completions.create(
                model=self.model_name, messages=messages, **generate_kwargs,
            )
            output = [choice.message.content for choice in chat_response.choices]
            if return_usage_tokens:
                return output[0], chat_response.usage.total_tokens
            return output[0]
            
        except Exception as e:
            traceback.print_exc()
            raise  # required for retry

    def generate_multiple(self, messages: List[Dict[str, str]], n: int = 8, max_n_per_call: int = 8, **kwargs) -> List[str]:
        generate_kwargs = self._get_updated_gen_params(**kwargs)
        outputs = []
        try:
            remaining = n
            while remaining > 0:
                batch = min(remaining, max_n_per_call)
                chat_response = self.client.chat.completions.create(
                    model=self.model_name, messages=messages, n=batch, **generate_kwargs,
                )
                outputs.extend(choice.message.content for choice in chat_response.choices)
                remaining -= batch
            return outputs

        except Exception as e:
            traceback.print_exc()
            import pdb;  pdb.set_trace()
            return ["<ERROR>"] * n

    def generate_w_logprob(self, messages: List[Dict[str, str]], **kwargs) -> str:
        generate_kwargs = self._get_updated_gen_params(**kwargs)
        try:
            chat_response = self.client.chat.completions.create(
                model=self.model_name, 
                messages=messages,
                logprobs=True,
                **generate_kwargs,
            )
            
            response = chat_response.choices[0].logprobs.content[0].token
            logprob = chat_response.choices[0].logprobs.content[0].logprob
            return response, logprob
            
        except Exception as e:
            traceback.print_exc()
            import pdb; pdb.set_trace()
            return {}

    def generate_structured(self, messages: List[Dict[str, str]], text_format: BaseModel, **kwargs) -> str:
        generate_kwargs = self._get_updated_gen_params(**kwargs)
        generate_kwargs = {k: v for k, v in generate_kwargs.items() if k != "max_completion_tokens"}
        try:
            response = self.client.responses.parse(
                model=self.model_name, input=messages, text_format=text_format, **generate_kwargs,
            )
            return response.output_parsed
        except Exception as e:
            traceback.print_exc()
            import pdb; pdb.set_trace()
            return "<ERROR>"
            

class GPT5Model(OpenAIModel):
    def _get_updated_gen_params(self, **kwargs):
        gen_kwargs = copy.deepcopy(self.gen_kwargs)
        gen_kwargs["max_output_tokens"] = kwargs.get("max_tokens", gen_kwargs.pop("max_tokens", 100))
        return gen_kwargs
    
    def generate(self, messages: List[Dict[str, str]], return_usage_tokens: bool=False, **kwargs) -> str:
        generate_kwargs = self._get_updated_gen_params(**kwargs)
        try:
            response = self.client.responses.create(
                model=self.model_name, input=messages, **generate_kwargs,
            )
            # Note: response has the following structure:
            # response.output[0] represents the reasoning object (which we don't use)
            #   - Example: ResponseReasoningItem(id='rs_0512159eeb7d25430069a86cc9c07c8190b34b0673aac8c5af', summary=[], type='reasoning', content=None, encrypted_content=None, status=None)
            # response.output[1] represents the output object
            #   - Example: ResponseOutputMessage(id='msg_0512159eeb7d25430069a86cc9e08c8190b158d4ab5763d208', content=[ResponseOutputText(annotations=[], text='<TEXT GENERATED BY THE MODEL>', type='output_text', logprobs=[])], role='assistant', status='completed', type='message')
            # In addition to ``output``, response has the following fields:
            # - parallel_tool_calls=True, 
            # - temperature=1.0, 
            # - tool_choice='auto', 
            # - tools=[],
            # - top_p=1.0, 
            # - usage=ResponseUsage(
            #   - input_tokens=76, input_tokens_details=InputTokensDetails(cached_tokens=0), 
            #   - output_tokens=40, 
            #   - output_tokens_details=OutputTokensDetails(reasoning_tokens=0), 
            #   - total_tokens=116)
            # ... and more
            output = response.output[1].content[0].text

            if return_usage_tokens:
                return output, response.usage.total_tokens
            return output
            
        except Exception as e:
            traceback.print_exc()
            import pdb;  pdb.set_trace()

            if return_usage_tokens:
                return "<ERROR>", generate_kwargs.get("max_output_tokens", 100)
            else:
                return "<ERROR>"



class HFModel(Model):
    def __init__(self, model_name: str, model_device: str, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.pad_token_id
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
        model.to(model_device)
        model.eval()

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        messages_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)
        output = self.model.generate(messages_ids, **self.gen_kwargs)[
            0, messages_ids.shape[1] :
        ]
        assist_out = self.tokenizer.decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return assist_out


def load_model(model_name: str, model_device: str, gen_kwargs: dict, **kwargs):
    if model_device in APIs:
        if model_name in ("gpt-5-nano", "gpt-5-nano-2025-08-07", "gpt-5-mini", "gpt-5-mini-2025-08-07"):
            return GPT5Model(model_name, gen_kwargs=gen_kwargs, **kwargs)
        else:
            return OpenAIModel(model_name, gen_kwargs=gen_kwargs, **kwargs)
    else:
        return HFModel(model_name, model_device, gen_kwargs=gen_kwargs, **kwargs)

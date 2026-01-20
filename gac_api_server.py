import argparse
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from utils.gac_gen_call import *
from utils.gac_gen_utils import *
import time
import uuid


@asynccontextmanager
async def lifespan(app: FastAPI):

    config_api_server, norm_type_api_server, threshold_api_server = load_yaml_config(
        args.config_path
    )

    global model_actors_list, tokenizers, vocab_union, mapping_matrices, index_to_vocab, special_prefix_tokens_dict, byte_mappings_list, min_max_position_embeddings, model_name_list, primary_index, threshold

    (
        model_actors_list,
        tokenizers,
        vocab_union,
        mapping_matrices,
        index_to_vocab,
        special_prefix_tokens_dict,
        byte_mappings_list,
        min_max_position_embeddings,
        model_name_list,
        primary_index,
        threshold,
    ) = setup_model_actors_and_data(
        config_api_server, norm_type_api_server, threshold_api_server
    )

    yield


parser = argparse.ArgumentParser(
    description="A script that uses a config file for GaC ensemble."
)
parser.add_argument(
    "--config-path",
    type=str,
    default="example_configs/example_thresholded_ensemble.yaml",
    help="Path to the configuration file.",
)
parser.add_argument(
    "--host",
    type=str,
    default="0.0.0.0",
    help="The host address to bind to. Default is 0.0.0.0",
)
parser.add_argument(
    "--port", type=int, default=8000, help="The port number to bind to. Default is 8000"
)
args = parser.parse_args()

app = FastAPI(lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    model: str = 'ensemble-model'
    messages: List[Message]
    max_length: Optional[int] = Field(default=None)  # Optional maximum length
    max_tokens: Optional[int] = Field(default=50)  # Specifying maximum new tokens
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    apply_chat_template: Optional[bool] = Field(default=True)
    # For early stopping
    until: Optional[List[str]] = Field(default=None)


@app.get("/status")
async def get_status():
    return {"status": "ready"}


@app.post("/v1/chat/completions")
async def api_generate(request: CompletionRequest):
    messages = [m.model_dump() for m in request.messages]
    max_length = request.max_length
    max_tokens = request.max_tokens
    temperature = request.temperature
    top_p = request.top_p
    apply_chat_template = request.apply_chat_template
    until = request.until

    kwargs = {}
    if max_length is not None:
        kwargs["max_length"] = max_length
    if max_tokens is not None:
        kwargs["max_new_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    
    logger.info(f"Arguments: {kwargs}")
    
    prepare_inputs = [
        model_actor.prepare_inputs_for_model.remote(
            messages, min_max_position_embeddings, apply_chat_template
        )
        for model_actor in model_actors_list
    ]
    models_inputs = ray.get(prepare_inputs)
    input_ids_0 = models_inputs[0]

    output = generate_ensemnble_response(
        model_actors_list=model_actors_list,
        model_name_list=model_name_list,
        tokenizers=tokenizers,
        vocab_union=vocab_union,
        mapping_matrices=mapping_matrices,
        index_to_vocab=index_to_vocab,
        special_prefix_tokens_dict=special_prefix_tokens_dict,
        byte_mappings_list=byte_mappings_list,
        primary_index=primary_index,
        threshold=threshold,
        until=until,
        **kwargs,
    )

    generated_texts = extract_generated_texts(tokenizers[0], input_ids_0, output)

    logger.info(f"Generated text:{generated_texts}")

    return {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_texts[0].strip()
                },
                "finish_reason": "length"
            }
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)

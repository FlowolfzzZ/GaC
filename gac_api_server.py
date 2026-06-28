import argparse
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from utils.gac_gen_call import *
from utils.gac_gen_utils import *
import asyncio
import time
import uuid

DEFAULT_MAX_TOKENS = 50
DEFAULT_GAC_MAX_BATCH_SIZE = 8
DEFAULT_GAC_BATCH_TIMEOUT = 2.0
DEFAULT_UNITE_MAX_BATCH_SIZE = 8
DEFAULT_UNITE_BATCH_TIMEOUT = 2.0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


@asynccontextmanager
async def lifespan(app: FastAPI):

    config_api_server, norm_type_api_server, threshold_api_server, ensemble_method, unite_top_k, unite_renorm, core_cfg = load_yaml_config(
        args.config_path
    )

    global model_actors_list, tokenizers, vocab_union, mapping_matrices
    global index_to_vocab, special_prefix_tokens_dict, byte_mappings_list
    global min_max_position_embeddings, model_name_list, primary_index, threshold
    global ensemble_method_g, unite_top_k_g, unite_renorm_g
    global id_to_str_list, str_to_ids_list, core_cfg_g
    global request_queue, max_batch_size_g, batch_timeout_g

    ensemble_method_g = ensemble_method
    unite_top_k_g = unite_top_k
    unite_renorm_g = unite_renorm
    core_cfg_g = core_cfg
    if args.max_batch_size is None:
        max_batch_size_g = (
            DEFAULT_UNITE_MAX_BATCH_SIZE
            if ensemble_method_g == "unite"
            else DEFAULT_GAC_MAX_BATCH_SIZE
        )
    else:
        max_batch_size_g = args.max_batch_size
    if args.batch_timeout is None:
        batch_timeout_g = (
            DEFAULT_UNITE_BATCH_TIMEOUT
            if ensemble_method_g == "unite"
            else DEFAULT_GAC_BATCH_TIMEOUT
        )
    else:
        batch_timeout_g = args.batch_timeout
    if ensemble_method_g == "unite":
        logger.info(
            "UNITE_CONFIG | algorithm=UniTE | selection=union-top-k "
            f"| top_k={unite_top_k_g} | renorm={unite_renorm_g} | config={args.config_path}"
        )
    else:
        logger.info(
            "GAC_CONFIG | algorithm=GaC | selection=full-union-vocab "
            f"| config={args.config_path}"
        )
    logger.info(
        "BATCH_CONFIG | "
        f"max_batch_size={max_batch_size_g} | batch_timeout={batch_timeout_g}"
    )
    request_queue = asyncio.Queue()

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
        id_to_str_list,
        str_to_ids_list,
    ) = setup_model_actors_and_data(
        config_api_server, norm_type_api_server, threshold_api_server, ensemble_method, core_cfg
    )

    worker_task = asyncio.create_task(batch_worker())

    yield

    worker_task.cancel()


async def batch_worker():
    loop = asyncio.get_running_loop()

    while True:
        batch: list[CompletionRequest] = []
        futures = []

        first_req, future = await request_queue.get()
        batch.append(first_req)
        futures.append(future)

        start_time = loop.time()

        while len(batch) < max_batch_size_g:
            remaining = batch_timeout_g - (loop.time() - start_time)
            if remaining <= 0:
                break
            try:
                req, future = await asyncio.wait_for(
                    request_queue.get(), timeout=remaining
                )
                batch.append(req)
                futures.append(future)
            except asyncio.TimeoutError:
                break

        logger.info(f"Processing batch of size {len(batch)}")

        batch_messages = [[m.model_dump() for m in req.messages] for req in batch]

        max_length = batch[0].max_length
        max_tokens = batch[0].max_tokens
        temperature = batch[0].temperature
        top_p = batch[0].top_p
        apply_chat_template = batch[0].apply_chat_template
        until = batch[0].until
        if max_length is None and max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS

        logger.info(
            "GENERATION_LIMITS | "
            f"batch_size={len(batch)} | max_length={max_length} | "
            f"max_new_tokens={max_tokens} | temperature={temperature} | top_p={top_p}"
        )

        kwargs = {}
        if max_length is not None:
            kwargs["max_length"] = max_length
        if max_tokens is not None:
            kwargs["max_new_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p

        try:
            prepare_inputs = [
                actor.prepare_inputs_for_model.remote(
                    batch_messages,
                    min_max_position_embeddings,
                    apply_chat_template,
                )
                for actor in model_actors_list
            ]
            models_inputs = ray.get(prepare_inputs)
            input_ids_0 = models_inputs[0]

            # 阻塞，防止请求消息卡住
            output = await loop.run_in_executor(
                None,
                lambda: generate_ensemnble_response(
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
                    ensemble_method=ensemble_method_g,
                    unite_top_k=unite_top_k_g,
                    unite_renorm=unite_renorm_g,
                    id_to_str_list=id_to_str_list,
                    str_to_ids_list=str_to_ids_list,
                    core_cfg=core_cfg_g,
                    **kwargs,
                ),
            )

            generated_texts = await loop.run_in_executor(
                None,
                lambda: extract_generated_texts(
                    tokenizers[0], input_ids_0, output
                ),
            )

            for future, text in zip(futures, generated_texts):
                if not future.done():
                    future.set_result(text.strip())

        except Exception as e:
            for future in futures:
                if not future.done():
                    future.set_exception(e)


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
parser.add_argument(
    "--max-batch-size",
    type=positive_int,
    default=None,
    help="Maximum queued requests to merge into one generation batch. Default: 1 for UniTE, 32 for GaC.",
)
parser.add_argument(
    "--batch-timeout",
    type=non_negative_float,
    default=None,
    help="Seconds to wait for more queued requests before running a batch. Default: 0 for UniTE, 2 for GaC.",
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
    max_tokens: Optional[int] = Field(default=DEFAULT_MAX_TOKENS)  # Specifying maximum new tokens
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
    future = asyncio.get_running_loop().create_future()
    await request_queue.put((request, future))

    generated_text = await future

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
                    "content": generated_text.strip()
                },
                "finish_reason": "stop"
            }
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)

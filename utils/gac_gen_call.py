from typing import Any, Dict, List, Set, Tuple

import ray
import yaml
import torch
from .ray_actor import get_remote_model_generator_class
from .gac_gen_utils import *
import numpy as np

def setup_model_actors_and_data(config: List[Dict], norm_type: str, threshold: float, ensemble_method: str = "gac", core_cfg: Dict = None) -> Tuple[List[Any], List[Any], Set[str], List[Dict[int, int]], Dict[int, str], Dict[Any, str], List[Dict[str, int]], int]:
    """
    Sets up model actors based on configurations and preprocesses necessary data for text generation.

    Args:
        config (List[Dict]): Configuration list where each element is a dictionary specifying
                             model path and memory specifications.
        norm_type (str): The type of normalization to apply ('average' or 'score').
        threshold (float)

    Returns:
        Tuple containing:
        - model_actors_list (List[ActorHandle]): List of Ray actor handles for the model generators.
        - tokenizers (List[Tokenizer]): List of tokenizer instances fetched from each model actor.
        - vocab_union (Set[str]): Unified set of all tokens across the tokenizers' vocabularies.
        - mapping_matrices (List[torch.sparse_coo_tensor]): A list of sparse COO tensors, each representing
          a mapping matrix from a model's tokenizer token IDs to the token IDs in the unified vocabulary.
          Each matrix corresponds to a tokenizer and maps its original token IDs to new token IDs in the
          unified vocabulary. The shape of each matrix is [model_vocab_size, len(vocab_union)], where
          model_vocab_size is the size of the tokenizer's vocabulary.
        - index_to_vocab (Dict[int, str]): Mapping from unique indices to tokens in the unified vocabulary.
        - special_prefix_tokens_dict (Dict[Tokenizer, str]): Mapping of each tokenizer to its special prefix token.
        - byte_mappings_list (List[Dict[str, int]]): List of byte value mappings for '<0x00>' to '<0xFF>'
          for each tokenizer.
        - min_max_position_embeddings (int): The minimum of the maximum position embeddings across all model actors.
        - model_name_list (List[str]): list of model name in model_actors_list
        - primary_index (int)
        - threshold (float)
        - id_to_str_list (List[Dict[int, str]]): per-model token id -> canonical union string (UniTE).
        - str_to_ids_list (List[Dict[str, List[int]]]): per-model canonical union string -> token ids (UniTE).
    """
    update_scores(config, norm_type)
    config = normalize_scores(config)
    logger.info(f"Model ensemble weights: {[(c['name'], round(c['score'],4)) for c in config]}")

    # find primary model
    primary_index = check_priorities(config)

    # Resolve the CoRE anchor ("main") model. CoRE here runs SYMMETRICALLY by default
    # (no anchor) since GaC fuses into a symmetric union vocab; this avoids overloading the
    # 'score' field and never forces a single model to dominate. An anchor (paper-faithful
    # asymmetric CoRE) is used only when explicitly requested via CORE_MAIN_INDEX, or when a
    # gate model is designated (priority: 'primary') — both are explicit choices, not 'score'.
    if core_cfg is not None and core_cfg.get('USE_CORE'):
        if core_cfg.get('main_index') is not None:
            core_main_index = core_cfg['main_index']
            if core_main_index >= len(config):
                raise ValueError(
                    f"CORE_MAIN_INDEX={core_main_index} is out of range for {len(config)} models."
                )
        elif primary_index != -1:
            core_main_index = primary_index
        else:
            core_main_index = None  # symmetric (no anchor)
        core_cfg['core_main_index'] = core_main_index
        if core_main_index is None:
            logger.info(
                f"CORE_CONFIG | enabled | mode=symmetric (no main) "
                f"| variant={core_cfg['variant']} | task_type={core_cfg['task_type']} "
                f"| hparams={core_cfg['hparams']}"
            )
        else:
            logger.info(
                f"CORE_CONFIG | enabled | mode=anchored "
                f"| main_model={config[core_main_index]['name']} (index {core_main_index}) "
                f"| variant={core_cfg['variant']} | task_type={core_cfg['task_type']} "
                f"| check_substring={core_cfg['check_substring']} | hparams={core_cfg['hparams']}"
            )
    if primary_index != -1:
        real_threshold = threshold*config[primary_index]["score"]
        logger.info(f"Gate model is {config[primary_index]['name']} with threshold {threshold}, and other ensembled models KV cache will be disabled!\nPlease note that for threshold ensemble, we currently only support batch size = 1.")

    else:
        real_threshold = threshold = 1
        logger.info("Ensemble mode: every-token | threshold=ignored")

    config = validate_and_update_quantization(config)

    # Initialize model actors based on configuration and GPU requirements
    model_actors_list = [
        get_remote_model_generator_class(model_config["num_gpus"]).remote(
            model_path=model_config["weight"], max_memory=model_config["max_memory"], model_name=model_config["name"], model_ensemble_weight=model_config["score"], use_cache=(primary_index == -1) or (i == primary_index), quantization=model_config["quantization"]
        )
        for i,model_config in enumerate(config)
    ]

    # Fetch tokenizer for each model
    tokenizers = [
        ray.get(model_actor.get_tokenizer.remote()) for model_actor in model_actors_list
    ]

    model_name_list = [
        ray.get(model_actor.get_model_name.remote()) for model_actor in model_actors_list
    ]

    # Determine special prefix tokens for all tokenizers
    special_prefix_tokens_dict = get_special_prefix_tokens_for_all(tokenizers)

    # Create a unified vocabulary and mappings for tokenizers
    (
        vocab_union,
        tokenizers_mapping,
        index_to_vocab,
        byte_mappings_list,
        id_to_str_list,
        str_to_ids_list,
    ) = get_vocab_union_and_mapping(tokenizers)

    model_vocab_size_list = [
        ray.get(model_actor.get_vocab_size.remote()) for model_actor in model_actors_list
    ]

    # UniTE works directly on each model's own top-k tokens and the lightweight string
    # maps above, so it does NOT need the GPU sparse projection matrices (nor the per-step
    # full-vocabulary sparse.mm). Only build them for GaC.
    if ensemble_method == "unite":
        mapping_matrices = None
    else:
        mapping_matrices = [
            create_mapping_matrix(mapping, len(vocab_union), vocab_size)
            for mapping, tokenizer, vocab_size in zip(tokenizers_mapping, tokenizers, model_vocab_size_list)
        ]

    # Find the minimum max position embeddings across all models
    min_max_position_embeddings = min(
        ray.get(model_actor.get_max_position_embeddings.remote())
        for model_actor in model_actors_list
    )

    return (
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
        real_threshold,
        id_to_str_list,
        str_to_ids_list,
    )

def validate_and_update_quantization(model_config: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Validates the 'quantization' field in each dictionary of a list of model configurations,
    and adds the 'quantization' field with a default value of 'none' if it's missing.

    Args:
        model_config (List[Dict[str, str]]): 
            A list of dictionaries, where each dictionary represents a model configuration.
            Each dictionary should contain a 'quantization' key, which must have one of the 
            following values: 'none', '8bit', or '4bit'. If the 'quantization' key is missing,
            it will be added with a default value of 'none'.

    Raises:
        ValueError: If any 'quantization' value is not one of 'none', '8bit', or '4bit'.

    Returns:
        List[Dict[str, str]]: The updated list of model configurations with valid 'quantization' values.
    """
    
    # Define the valid quantization options
    valid_quantization_values = {'none', '8bit', '4bit'}
    
    # Loop through each configuration in the input list
    for idx, config in enumerate(model_config):
        # Check if 'quantization' key exists, if not, set it to 'none'
        if 'quantization' not in config:
            config['quantization'] = 'none'
        
        # Get the 'quantization' value
        quantization_value = config['quantization']
        
        # Check if the value is valid, otherwise raise an error with details
        if quantization_value not in valid_quantization_values:
            raise ValueError(
                f"Invalid quantization value '{quantization_value}' in config at index {idx}. "
                f"Allowed values are: {valid_quantization_values}"
            )
    
    # Return the updated list of configurations
    return model_config

def check_priorities(dict_list):
    """
    Check the list of dictionaries to ensure that there is exactly one "primary" priority and all priorities are valid.

    Args:
    dict_list (list of dict): A list where each item is a dictionary with a key "priority" whose value should be either "supportive" or "primary".

    Returns:
    int: Index of the first dictionary with "primary" as priority if there is exactly one, otherwise returns -1.
    """
    allowed_priorities = ["supportive", "primary"]
    primary_index = -1
    primary_count = 0

    for index, d in enumerate(dict_list):
        priority = d.get("priority")

        # Check if the priority is within the allowed values
        if priority not in allowed_priorities:
            raise ValueError(f"'priority' value '{priority}' at index {index} is not allowed!")

        # Check for primary priority and count them
        if priority == "primary":
            primary_count += 1
            if primary_count == 1:
                primary_index = index

    # Warn if there is more than one primary priority
    if primary_count > 1:
        raise ValueError("More than one 'primary' found!")

    return primary_index


def normalize_scores(config, n=1):
    """
    Normalizes the scores of each configuration in the list of dictionaries by multiplying each score by n,
    and then normalizing these scores to a 0 to 1 range such that their sum is 1.
    
    Parameters:
        config (list of dict): A list of dictionaries, each representing a configuration with a 'score' key.
        n (int, optional): The factor to multiply each score by before normalization. Defaults to 1.
    
    Returns:
        list of dict: The input list of dictionaries with normalized 'score' values.
    """
    
    # Extract scores and multiply by n
    scores = np.array([configuration['score'] for configuration in config]) ** n
    
    # Normalize scores to sum to 1
    normalized_scores = scores / np.sum(scores)
    
    # Update the scores in the original list of dictionaries
    for configuration, new_score in zip(config, normalized_scores):
        configuration['score'] = new_score
    
    return config

def extract_generated_texts(tokenizer, input_ids_0: torch.Tensor, output: torch.Tensor) -> List[str]:
    """
    Extract generated text from the model's output, excluding the input portion and any left-side padding.

    :param tokenizer: The tokenizer used, which must have a pad_token_id attribute.
    :param input_ids_0: Token IDs input to the model, shaped (batch_size, sequence_length).
                        Input may contain left-side padding.
    :param output: Model output token IDs, shaped (batch_size, output_sequence_length).
                Output sequence contains both the input sequence and the generated response.
    :return: A list of strings, where each string is the generated text for the corresponding batch.

    Function logic:
    - For each sample, find the non-pad portion in input_ids_0.
    - Search for a matching sequence in the output that corresponds to the non-pad portion.
    - Extract from the end of the matched sequence in the output to the end of the output as the response.
    - Decode the token IDs of the response into text using the tokenizer.
    """
    pad_token_id = tokenizer.pad_token_id
    generated_texts = []

    for i in range(output.shape[0]):
        # Find the index of the first non-pad token in input_ids_0
        non_pad_indices = (input_ids_0[i] != pad_token_id).nonzero().squeeze()
        
        if non_pad_indices.dim() == 0:
            non_pad_indices = non_pad_indices.unsqueeze(0)

        first_non_pad_index = non_pad_indices[0].item() if non_pad_indices.numel() > 0 else -1

        if first_non_pad_index == -1:
            raise ValueError("No non-pad tokens found in the input for batch index {}".format(i))

        # Construct the input_ids tensor of the non-pad portion for the current sample
        input_ids_non_pad = input_ids_0[i, first_non_pad_index:]

        found_match = False
        for pos in range(output.shape[1]):
            if pos + input_ids_non_pad.shape[0] <= output.shape[1]:
                if torch.equal(output[i, pos:pos+input_ids_non_pad.shape[0]], input_ids_non_pad):
                    found_match = True
                    response_start_index = pos + input_ids_non_pad.shape[0]
                    break

        if not found_match:
            raise ValueError(f"No matching sequence found in the output for batch index {i}")

        response_ids = output[i, response_start_index:]

        decoded_text = tokenizer.decode(response_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        generated_texts.append(decoded_text)

    return generated_texts

def update_scores(config, norm_type):
    """
    This function updates each dictionary in a list by different strategies based on the norm_type value.
    - 'average': Sets all scores to 1.
    - 'score': Leaves the "score" values unchanged.
    
    If the norm_type is not one of the specified values, an error is raised.

    Parameters:
    - config (list of dict): A list of dictionaries, each containing the fields "score" and "ece".
    - norm_type (str): The type of normalization to apply ('average' or 'score').

    Returns:
    - The updated list of dictionaries according to the specified normalization type.
    
    Raises:
    - ValueError: If norm_type is not one of the specified values.
    """
    if norm_type == 'average':
        for item in config:
            item["score"] = 1
    elif norm_type == 'score':
        pass
    else:
        raise ValueError(f"Invalid norm_type: {norm_type}. Expected 'average' or 'score'.")

    return config

def load_yaml_config(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)

    try:
        config_api_server = config['CONFIG_API_SERVER']
        norm_type_api_server = config['NORM_TYPE_API_SERVER']
        threshold_api_server = config['THRESHOLD_API_SERVER']
    except KeyError as e:
        raise ValueError(f"Missing required configuration key: {e}")

    # Optional switch to select the ensembling algorithm. Defaults to 'gac' so
    # existing configs (without these keys) keep the original GaC behavior.
    ensemble_method = config.get('ENSEMBLE_METHOD', 'gac')
    valid_ensemble_methods = {'gac', 'unite'}
    if ensemble_method not in valid_ensemble_methods:
        raise ValueError(
            f"Invalid ENSEMBLE_METHOD: '{ensemble_method}'. Expected one of {valid_ensemble_methods}."
        )

    # Top-k used by UniTE (ignored by GaC). Default 10 follows the UniTE paper.
    unite_top_k = config.get('UNITE_TOP_K', 10)
    if not isinstance(unite_top_k, int) or unite_top_k <= 0:
        raise ValueError(f"Invalid UNITE_TOP_K: '{unite_top_k}'. Expected a positive integer.")

    # Whether UniTE renormalizes each model's probabilities over the candidate set.
    # Defaults to True: the faithful UniTE behavior — each model's probabilities are
    # renormalized over the candidate set S and then combined as a weighted average.
    # Set to False to instead keep the absolute (model-weighted) probabilities and sum
    # them like GaC (a "GaC restricted to the candidate set" variant). Ignored by GaC.
    unite_renorm = config.get('UNITE_RENORM', True)
    if not isinstance(unite_renorm, bool):
        raise ValueError(f"Invalid UNITE_RENORM: '{unite_renorm}'. Expected a boolean.")

    # Optional CoRE augmentation (paper: "Harnessing Consistency for Robust Test-Time LLM
    # Ensemble"). CoRE rides on top of the GaC union-vocab path: it harnesses token- and
    # model-level consistency to robustly reweight the ensemble. All keys are optional;
    # when USE_CORE is False (the default) the GaC/UniTE behaviour is completely unchanged.
    core_cfg = parse_core_config(config, ensemble_method)

    return (
        config_api_server,
        norm_type_api_server,
        threshold_api_server,
        ensemble_method,
        unite_top_k,
        unite_renorm,
        core_cfg,
    )


def parse_core_config(config, ensemble_method):
    """
    Parse and validate the optional CoRE-related keys from a loaded YAML config.

    Returns a dict with the resolved CoRE settings. ``main_index`` holds the user-provided
    override (or None). The concrete anchor is resolved later in
    ``setup_model_actors_and_data``: CoRE runs symmetrically (no main) by default, and only
    uses an anchor when CORE_MAIN_INDEX is set or a gate model (priority: 'primary') exists.
    """
    use_core = config.get('USE_CORE', False)
    if not isinstance(use_core, bool):
        raise ValueError(f"Invalid USE_CORE: '{use_core}'. Expected a boolean.")

    valid_variants = {
        'consist-rbf', 'consist-power', 'consist-sig', 'consist-linear', 'consist-rec'
    }
    variant = config.get('CORE_VARIANT', 'consist-rbf')
    if variant not in valid_variants:
        raise ValueError(
            f"Invalid CORE_VARIANT: '{variant}'. Expected one of {valid_variants}."
        )

    valid_task_types = {'generation', 'classification'}
    task_type = config.get('CORE_TASK_TYPE', 'generation')
    if task_type not in valid_task_types:
        raise ValueError(
            f"Invalid CORE_TASK_TYPE: '{task_type}'. Expected one of {valid_task_types}."
        )

    main_index = config.get('CORE_MAIN_INDEX', None)
    if main_index is not None and (not isinstance(main_index, int) or main_index < 0):
        raise ValueError(
            f"Invalid CORE_MAIN_INDEX: '{main_index}'. Expected a non-negative integer."
        )

    check_substring = config.get('CORE_CHECK_SUBSTRING', True)
    if not isinstance(check_substring, bool):
        raise ValueError(
            f"Invalid CORE_CHECK_SUBSTRING: '{check_substring}'. Expected a boolean."
        )

    # CoRE here augments only the union-vocab GaC path; it is not implemented for UniTE's
    # candidate-set fusion, so reject that combination early with a clear message.
    if use_core and ensemble_method == 'unite':
        raise ValueError(
            "USE_CORE is only supported with the GaC ensemble method, not UniTE. "
            "Set ENSEMBLE_METHOD to 'gac' (or omit it) to use CoRE."
        )

    # Consistency-function hyperparameters (paper defaults).
    hparams = {
        'beta': float(config.get('CORE_BETA', 2.0)),    # consist-rbf
        'alpha': float(config.get('CORE_ALPHA', 5.0)),  # consist-power
        'gamma': float(config.get('CORE_GAMMA', 1.0)),  # consist-rec
        'k': float(config.get('CORE_K', 5.0)),          # consist-sig
    }

    return {
        'USE_CORE': use_core,
        'variant': variant,
        'task_type': task_type,
        'main_index': main_index,        # user override (or None -> auto)
        'core_main_index': None,         # resolved anchor, filled in by setup
        'check_substring': check_substring,
        'hparams': hparams,
    }

# init RAY
ray.init()

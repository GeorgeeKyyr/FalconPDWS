import argparse
import hashlib
import logging
import math
import os
import pickle
import random
import time
from typing import Any

import numpy as np
import torch
from petlib.pack import decode, encode
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import crypto
from reduce import (
    get_probability_distribution_for_bit,
    sample_bit,
    sample_bit_by_sample_type,
    simple_encoder,
)

MAX_TIME_BEFORE_PLANT_ERROR = 300  # seconds
STOP_TOKEN = "</s>"

logging.basicConfig(filename="logging.log", encoding="utf-8", level=logging.INFO)

def main(args: argparse.Namespace) -> None:
    generated_text = generate_text(args)
    print(generated_text, file=open("wat.txt", "w"))

def generate_text(args: argparse.Namespace) -> str:
    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype="auto",
        load_in_4bit=args.load_in_4bit,
    )
    logging.info(f"loaded weights in {model.config.torch_dtype}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        max_length=model.config.max_position_embeddings,
        truncation=True,
    )

    if args.gen_type == "plain":
        generated_text, _ = generate_text_plain(args.prompt, args.num_tokens, model, tokenizer, args.sample_type)
    elif args.gen_type == "plain_with_bits":
        generated_text, _, _ = generate_text_plain_with_bits(args.prompt, args.num_tokens, model, tokenizer, args.sample_type)
    elif args.gen_type == "asymmetric":
        generated_text, _, pk, params, _, _ = generate_text_asymmetric(
            args.prompt,
            model,
            tokenizer,
            args.sample_type,
            args.message_length,
            args.signature_segment_length,
            args.bit_size,
            args.max_planted_errors,
            args.sk,
            args.pk,
            args.params,
            args.continue_until_stop_token,
        )
    elif args.gen_type == "symmetric":
        generated_text, _, _ = generate_text_symmetric(
            args.prompt,
            args.num_tokens,
            model,
            tokenizer,
            args.sample_type,
            args.security_parameter,
        )
    else:
        raise ValueError(f"Unsupported gen_type: {args.gen_type}")

    return generated_text

def generate_text_plain(prompt, num_tokens, model, tokenizer, sample_type):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    initial_len = inputs.shape[1]
    attn = torch.ones_like(inputs)
    past = None
    vocab_size = len(tokenizer)

    for _ in tqdm(range(num_tokens)):
        token, inputs, past, attn = sample_token(model, tokenizer, inputs, past, attn, vocab_size, sample_type)

    output_tokens = inputs[:, initial_len:]
    output_text = tokenizer.decode(output_tokens.squeeze(), skip_special_tokens=True)
    return output_text, output_tokens

def generate_text_plain_with_bits(prompt, num_tokens, model, tokenizer, sample_type):
    encode, decode_fn, padded_encoding, max_bit_length = simple_encoder(tokenizer.get_vocab().values())
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    initial_len = inputs.shape[1]
    attn = torch.ones_like(inputs)
    past = None
    vocab_size = len(tokenizer)
    bitstring = ""

    for _ in tqdm(range(num_tokens)):
        with torch.no_grad():
            if past:
                output = model(inputs[:, -1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)
        probs = torch.softmax(output.logits[:, -1, :vocab_size], dim=-1).cpu().numpy().squeeze()
        bits = ""
        for j in range(max_bit_length):
            bit = sample_bit(probs, padded_encoding, j, max_bit_length, bits, sample_type)
            bits += str(bit)
        token = decode_fn(bits)
        inputs = torch.cat([inputs, torch.tensor([[token]]).to(model.device)], dim=-1)
        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)
        bitstring += bits

    output_tokens = inputs[:, initial_len:]
    output_text = tokenizer.decode(output_tokens.squeeze(), skip_special_tokens=True)
    return output_text, output_tokens, bitstring

def generate_text_symmetric(prompt, num_tokens, model, tokenizer, sample_type, security_parameter):
    encode, decode_fn, padded_encoding, max_bit_length = simple_encoder(tokenizer.get_vocab().values())
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    initial_len = inputs.shape[1]
    attn = torch.ones_like(inputs)
    past = None
    vocab_size = len(tokenizer)
    entropy = 0.0
    bitstring = ""
    r = ""

    for i in tqdm(range(num_tokens)):
        with torch.no_grad():
            if past:
                output = model(inputs[:, -1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)
        probs = torch.softmax(output.logits[:, -1, :vocab_size], dim=-1).cpu().numpy().squeeze()
        bits = ""
        prev_bits = ""
        for j in range(max_bit_length):
            pr = get_probability_distribution_for_bit(probs, padded_encoding, j, max_bit_length, prev_bits)
            if entropy < security_parameter:
                bit = sample_bit_by_sample_type(sample_type, pr)
                entropy -= math.log2(pr[bit])
                if entropy >= security_parameter:
                    r = bitstring + str(bit)
            else:
                hash_index = i * max_bit_length + j
                unkeyed_hash = crypto.unkeyed_hash_to_float(bytes(r, "utf-8") + bytes(bin(hash_index), "utf-8"))
                bit = 1 if unkeyed_hash <= pr[1] else 0
            bits += str(bit)
            prev_bits = bits
        token = decode_fn(bits)
        inputs = torch.cat([inputs, torch.tensor([[token]]).to(model.device)], dim=-1)
        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)
        bitstring += bits

    output_tokens = inputs[:, initial_len:]
    output_text = tokenizer.decode(output_tokens.squeeze(), skip_special_tokens=True)
    return output_text, output_tokens, bitstring

def generate_text_asymmetric(prompt, model, tokenizer, sample_type, message_length, signature_segment_length, bit_size, max_planted_errors, sk_path, pk_path, params_path, continue_until_stop_token):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    initial_len = inputs.shape[1]
    attn = torch.ones_like(inputs)
    past = None

    if sk_path and os.path.exists(sk_path) and pk_path and os.path.exists(pk_path) and params_path and os.path.exists(params_path):
        logging.info("loading Falcon keys from disk")
        with open(sk_path, "rb") as f:
            sk = decode(pickle.load(f))
        with open(pk_path, "rb") as f:
            pk = decode(pickle.load(f))
        with open(params_path, "rb") as f:
            params = decode(pickle.load(f))
    else:
        logging.info("generating new Falcon keys")
        sk, pk, params = crypto.falcon_generate()
        if sk_path:
            with open(sk_path, "wb") as f:
                pickle.dump(encode(sk), f)
        if pk_path:
            with open(pk_path, "wb") as f:
                pickle.dump(encode(pk), f)
        if params_path:
            with open(params_path, "wb") as f:
                pickle.dump(encode(params), f)

    message = prompt.encode("utf-8")
    message_hash = hashlib.sha256(message).digest()
    signature = crypto.sign_and_encode_openssl(sk, message_hash, params, max_planted_errors)
    logging.info(f"signature: {signature}")

    # Placeholder: actual asymmetric embedding logic goes here
    # For now we just return the prompt + fake signature string
    generated_text = prompt + "\n[FALCON_SIG] " + signature[:64] + "..."
    generated_tokens = tokenizer.encode(generated_text, return_tensors="pt")
    return generated_text, generated_tokens, pk, params, 0, 0


def sample_token(model, tokenizer, inputs, past, attn, vocab_size, sample_type, embedded_first=False, top_p=0.9, temperature=0.9):
    sampled = False
    while not sampled:
        with torch.no_grad():
            if past:
                output = model(inputs[:, -1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)
            logits = output.logits[:, -1, :vocab_size]

            if sample_type == "argmax":
                token = torch.argmax(logits, dim=-1, keepdim=True)
            elif sample_type == "multinomial":
                probs = torch.softmax(logits, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
            elif sample_type == "nucleus":
                sorted_logits, sorted_indices = torch.sort(logits / temperature, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[..., indices_to_remove] = float("-inf")
                probs = torch.softmax(logits, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
            else:
                raise ValueError(f"Unsupported sample_type: {sample_type}")

            token_str = tokenizer.decode(token.squeeze().cpu())
            if not embedded_first and STOP_TOKEN in token_str:
                continue  # try again
            sampled = True

            token = token.to(inputs.device)
            inputs = torch.cat([inputs, token.unsqueeze(0)], dim=-1)
            past = output.past_key_values
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    return token, inputs, past, attn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="There once was a", type=str)
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1", type=str)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction)
    parser.add_argument("--num-tokens", default=80, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gen-type", default="asymmetric", choices=["plain", "plain_with_bits", "symmetric", "asymmetric"])
    parser.add_argument("--sample-type", default="multinomial", choices=["argmax", "multinomial", "nucleus"])
    parser.add_argument("--sk", default="sk.pickle", type=str)
    parser.add_argument("--pk", default="pk.pickle", type=str)
    parser.add_argument("--params", default="params.pickle", type=str)
    parser.add_argument("--message-length", default=crypto.DEFAULT_MESSAGE_LENGTH, type=int)
    parser.add_argument("--signature-segment-length", default=crypto.DEFAULT_SIGNATURE_SEGMENT_LENGTH, type=int)
    parser.add_argument("--bit-size", default=crypto.DEFAULT_BIT_SIZE, type=int)
    parser.add_argument("--max-planted-errors", default=crypto.DEFAULT_MAX_PLANTED_ERRORS, type=int)
    parser.add_argument("--continue-until-stop-token", action=argparse.BooleanOptionalAction)
    parser.add_argument("--security-parameter", default=crypto.DEFAULT_SECURITY_PARAMETER, type=int)
    main(parser.parse_args())

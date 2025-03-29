import hashlib
from struct import unpack
from typing import Any
from bitstring import BitArray
from reedsolo import RSCodec

# Falcon implementation from pqcrypto
from falcon import generate_keypair, sign, verify

# Constants
SIGNATURE_LENGTH: int = 666  # Falcon-512 signatures are ~666 bytes
REED_SOLO_CONSTANT: int = 8

DEFAULT_SIGNATURE_SEGMENT_LENGTH = 16
DEFAULT_BIT_SIZE = 2
DEFAULT_MESSAGE_LENGTH = DEFAULT_SIGNATURE_SEGMENT_LENGTH // DEFAULT_BIT_SIZE
DEFAULT_MAX_PLANTED_ERRORS = 2
DEFAULT_SECURITY_PARAMETER = 16

def get_signature_codeword_length(max_planted_errors: int, bit_size: int) -> int:
    codeword_length = SIGNATURE_LENGTH + (REED_SOLO_CONSTANT * max_planted_errors * 2)
    assert codeword_length % bit_size == 0
    return codeword_length

def unkeyed_hash_to_float(input_bytes: bytes) -> float:
    return float(unpack("L", hashlib.sha256(input_bytes).digest()[:8])[0]) / 2**64

def unkeyed_hash_to_bits(input_bytes: bytes, bit_size: int) -> str:
    assert bit_size <= 256
    return BitArray(bytes=hashlib.sha256(input_bytes).digest()).bin[:bit_size]

def bytes_to_binary_codeword(input_bytes: bytes, max_planted_errors: int) -> str:
    if max_planted_errors == 0:
        return BitArray(bytes=input_bytes).bin
    rsc = RSCodec(max_planted_errors * 2)
    return BitArray(bytes=rsc.encode(input_bytes)).bin

def binary_codeword_to_bytes(binary_codeword: str, max_planted_errors: int) -> bytes:
    if max_planted_errors == 0:
        return BitArray(bin=binary_codeword).bytes
    rsc = RSCodec(max_planted_errors * 2)
    return rsc.decode(BitArray(bin=binary_codeword).bytes)[0]

# FALCON FUNCTIONS
def falcon_generate() -> tuple[list, bytes, tuple]:
    pk, sk = generate_keypair()
    return [sk], pk, ()

def falcon_sign(message: bytes, sk: list, params: tuple) -> bytes:
    return sign(message, sk[0])

def falcon_verify(message: bytes, signature: bytes, pk: bytes, params: tuple) -> bool:
    try:
        verify(message, signature, pk)
        return True
    except Exception:
        return False

def sign_and_encode_openssl(sk: list, message: bytes, params: tuple, max_planted_errors: int) -> str:
    signature = falcon_sign(message, sk, params)
    h = hashlib.sha512(message).digest()
    codeword = bytes_to_binary_codeword(signature, max_planted_errors)
    return BitArray(
        bytes=bytes(a ^ b for a, b in zip(BitArray(bin=codeword).bytes, h, strict=False))
    ).bin

def decode_and_verify_openssl(pk: bytes, message: bytes, binary_codeword: str, params: tuple, max_planted_errors: int) -> bool:
    h = hashlib.sha512(message).digest()
    unmasked = BitArray(
        bytes=bytes(a ^ b for a, b in zip(BitArray(bin=binary_codeword).bytes, h, strict=False))
    ).bin
    signature = binary_codeword_to_bytes(unmasked, max_planted_errors)
    return falcon_verify(message, signature, pk, params)

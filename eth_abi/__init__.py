from importlib.metadata import (
    version as __version,
)

from eth_abi.abi import (
    decode,
    encode,
    is_encodable,
    is_encodable_type,
    resolve_hooks,
)
from eth_abi.hooks import (
    EncodingContext,
)

__version__ = __version("eth-abi")

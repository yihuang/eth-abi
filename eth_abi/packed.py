from .codec import (
    ABIEncoder,
)
from .registry import (
    registry_packed,
)

default_encoder_packed = ABIEncoder(registry_packed, is_packed=True)

encode_packed = default_encoder_packed.encode
is_encodable_packed = default_encoder_packed.is_encodable
resolve_hooks_packed = default_encoder_packed.resolve_hooks

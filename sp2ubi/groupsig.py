"""Group-signature interface. The implementation used by the protocol is BBS04 over BLS12-381 (groupsig_bbs04.py)."""
from abc import ABC, abstractmethod


class GroupSignature(ABC):
    @abstractmethod
    def sign(self, member_key, msg: bytes) -> bytes: ...

    @abstractmethod
    def verify(self, msg: bytes, sig: bytes) -> bool: ...

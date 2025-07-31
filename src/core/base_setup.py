from enum import IntEnum

__all__ = [
    "DISPLAY_TO_HEX",
    "HEX_DISPLAY_MAP",
    "HexSymbol",
    "mapper_to_int",
    "mapper_to_str",
]


# due to using numpy.int8 as medium there can't be more than 127 exiting symbols...
# But i dont think it is gonna be a problem... right..?
class HexSymbol(IntEnum):
    """
    Represents supported hex symbols alongside with some control values.
    
    .. note::
        The values are chosen to be compatible with numpy.int8 and should not exceed its range (-128 to 127).
    """

    S_STOP = -1
    """
    Control value signalizing end of sequence, used for padding in some solvers.
    """
    S_BLANK = 0
    """
    Blank symbol used for displaying solutions and buffer.
    """
    # base game
    S_1C = 1
    S_55 = 2
    S_BD = 3
    S_E9 = 4
    S_7A = 5
    S_FF = 6
    # dlc
    S_X9 = 7
    S_XX = 8
    S_XH = 9
    S_IX = 10
    S_XR = 11


HEX_DISPLAY_MAP: dict[HexSymbol, str] = {
    HexSymbol.S_STOP: "??",
    HexSymbol.S_BLANK: " ▧",
    HexSymbol.S_1C: "1C",
    HexSymbol.S_55: "55",
    HexSymbol.S_BD: "BD",
    HexSymbol.S_E9: "E9",
    HexSymbol.S_7A: "7A",
    HexSymbol.S_FF: "FF",
    HexSymbol.S_X9: "X9",
    HexSymbol.S_XX: "XX",
    HexSymbol.S_XH: "XH",
    HexSymbol.S_IX: "IX",
    HexSymbol.S_XR: "XR",
}

DISPLAY_TO_HEX: dict[str, HexSymbol] = {v: k for k, v in HEX_DISPLAY_MAP.items()}


mapper_to_str = HEX_DISPLAY_MAP.__getitem__
mapper_to_int = DISPLAY_TO_HEX.__getitem__

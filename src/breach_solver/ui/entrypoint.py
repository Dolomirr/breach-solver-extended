#!/home/dolomirr/Projects/breach-solver-extended/.venv/bin/python3
# temporary shebang for testing
import argparse
import sys

_SOLVER_MAP = {
    "antcol": "antcol",
    "ac": "antcol",
    "bruter": "bruter",
    "br": "bruter",
    "linear": "linear",
    "ln": "linear",
    "auto": "auto",
}


DEFAULT_OPEN_BROWSER = "__DEFAULT_OPEN_BROWSER__"


def solver_type(val: str) -> str:
    key = val.lower()
    if key in _SOLVER_MAP:
        return _SOLVER_MAP[val]
    msg = f"invalid solver '{val}'. Valid options: " + r"{" + f"{', '.join(sorted(set(_SOLVER_MAP.keys())))}" + r"}"
    raise argparse.ArgumentTypeError(msg)


def run_gui_window(*args, **kwargs) -> int:
    """Temp placeholder"""
    print("running in window...")
    print(args, kwargs)
    return 0


def run_gui_browser(*args, **kwargs) -> int:
    """Temp placeholder"""
    print("Running in browser...")
    print(args, kwargs)
    return 0


def run_cli(*args, **kwargs) -> int:
    """Temp placeholder"""
    print("Solving in console...")
    print(*args, **kwargs)
    return 0

def  _split_file_tokens(files_list):
    out = []
    if not files_list:
        return out
    for sub in files_list:
        if not isinstance(sub, list):
            out.append(sub)
        else:
            out.extend(sub)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="Breach Solver Extended",
        description="Solver for 'Breach Protocol' mini game with additional and non-vanilla feature support.",
        epilog=(
            "Examples:\n"
            + "    main                            # opens gui in stand-alone window\n"
            + "    main -s ln screenshot.png       # solves from screenshot.png using linear solver\n"
            + "    main -g -b file1.png file2.ong  # open gui in browser and preloads file1.png !(only first one will be used)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode_group = p.add_mutually_exclusive_group()

    mode_group.add_argument(
        "-g",
        "--gui",
        action="store_true",
        help="Launch GUI in native window mode, mutually exclusive with '-b/--browser'.",
    )

    mode_group.add_argument(
        "-b",
        "--browser",
        action="store_true",
        help="Launch GUI in browser (on localhost), mutually exclusive with '-g/--gui'.",
    )

    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for browser GUI.",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for browser GUI",
    )
    p.add_argument(
        "--open",
        nargs="?",
        const=DEFAULT_OPEN_BROWSER,
        help=(
            "Auto-open browser. If given without value open in system default browser. \n"
            + "Optionally allow to provide executable/command (e.g. '/usr/bin/firefox') "
            + "or 'C:\\Program Files\\Mozilla Firefox\\firefox.exe' to run in specified browser. "
        ),
    )

    p.add_argument(
        "-s",
        "--solver",
        type=solver_type,
        default="auto",
        metavar="SOLVER",
        help=(
            "{antcol (ac), bruter (br), linear (ln), auto}.\n"
            + "Allow to provide solver to use, it set to 'auto' solver will automatically decide which "
            + "method to use depending on problem complexity."
        ),
    )

    # TODO: decide where to place config file
    p.add_argument(
        "-c",
        "--config",
        metavar="FILE",
        default=None,
        help="Path to config file (optional).",
    )

    p.add_argument(
        "-f",
        "--files",
        nargs="+",
        action="append",
        help="Files to process (can accept multiple times), tokens may be separated by commas.",
    )

    p.add_argument(
        "positional",
        nargs="*",
        help="Input file(s) to solve or (if --gui used) preload list will be merged with '-f/--files'.",
    )
    return p


def main(argv=None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    files = _split_file_tokens(args.files) + list(args.positional)

    from icecream import ic  # noqa: PLC0415
    ic(files)
    
    if args.gui or args.browser:
        preload = None
        if len(args.files) > 1:
            preload = files
            parser.error(f"Warning: '--gui' and '--browser' accepts at most one file to preload, only the first file will be used ({preload}).")
        
        if args.browser:
            run_gui_browser(file=preload, host=args.host, port=args.port, open_in=args.open)
        else:
            run_gui_window(file=preload, host=args.host, port=args.port, open_in=args.open)
        return

    run_cli(files, args.config)
    return
    

if __name__ == "__main__":
    main()

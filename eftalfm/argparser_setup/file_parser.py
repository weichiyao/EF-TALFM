import argparse
import shlex


class ArgumentFileParser(argparse.ArgumentParser):
    """Allow `@file.args` experiment files with shell-like splitting and comments."""

    def convert_arg_line_to_args(self, arg_line: str):
        line = arg_line.strip()
        if (not line) or line.startswith("#"):
            return []
        lexer = shlex.shlex(line, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = "#"
        return list(lexer)

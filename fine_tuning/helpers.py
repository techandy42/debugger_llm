def indent_lines(string: str) -> str:
  indented_string = '\n'.join('    ' + line for line in string.splitlines())
  return indented_string

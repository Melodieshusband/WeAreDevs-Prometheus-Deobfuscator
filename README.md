# Created by: Melodie's husband (Meloten)
<div align="center">

# WeAreDevs Deobfuscator

Static deobfuscator for Lua / Luau scripts protected by WeAreDevs and Prometheus-style obfuscators.

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Dependencies](https://img.shields.io/badge/dependencies-none-success)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Status](https://img.shields.io/badge/status-WIP-orange)

</div>

## Features

- Executes nothing: works purely on text and AST (lexer, parser, tree transforms)
- Decodes the string layer and folds constant arithmetic across the whole tree
- Locates the VM dispatcher and breaks it into blocks with their transitions
- Builds a block graph and finds function entry points
- Recovers `if` / `while` structure and lifts closures back into plain functions
- Zero external dependencies

## Installation

```bash
git clone https://github.com/Melodieshusband/WeAreDevs-deobfuscator.git
cd WeAreDevs-deobfuscator
```

Only Python 3.10+ is required. Optionally, install it as a package:

## Usage

```bash
python deobfuscate.py script.lua
python deobfuscate.py a.lua b.lua -o result
python -m wearedevs_deobfuscator script.lua --no-blocks
```

| Flag | Description |
|------|-------------|
| `-o`, `--output-dir` | output directory (default: `output`) |
| `--no-blocks` | skip the VM dispatcher analysis |

## Output

For every input file the tool writes:

| File | Contents |
|------|----------|
| `<name>.decoded.lua` | strings decoded, constants folded |
| `<name>.clean.lua` | parsed into an AST and reformatted |
| `<name>.blocks.txt` | VM dispatcher blocks with entry conditions and transitions |
| `<name>.cfg.txt` | block graph, functions and their entry points |
| `<name>.structured.lua` | functions with `if` / `while` recovered |
| `<name>.lifted.lua` | closures lifted into regular functions |
| `<name>.script.lua` | final cleaned-up script |

## How it works

```
source.lua
   │
   ▼
lexer → parser → AST
   │
   ▼
strings layer → constant folding
   │
   ▼
dispatcher → transitions → CFG
   │
   ▼
structure → lift → cleanup → script.lua
```

## Project layout

```
deobfuscate.py                  entry point
wearedevs_deobfuscator/
    lexer.py, parser.py         Lua tokenizer and parser
    ast_nodes.py, visitor.py    AST and tree traversal
    fold.py, transform.py       constant folding
    wearedevs.py                WeAreDevs string layer
    strings_layer.py            string decoding
    dispatcher.py               VM dispatcher detection
    transitions.py              transitions between blocks
    graph.py                    block graph
    structure.py                if / while recovery
    lift.py                     closure lifting
    constants.py                constant decryption
    peephole.py, simplify.py    expression simplification
    cleanup.py, clean.py        final cleanup
    roles.py                    variable roles
    printer.py                  AST to Lua
    pipeline.py, cli.py         pipeline and command line
```

## Limitations

- Some transitions (reported as `dynamic`) go through `U["key"]` lookups and are not resolved yet
- Not every function can be structured; those remain as a state machine
- Tuned for WeAreDevs-style output, other obfuscators may not work

## Disclaimer

This tool is intended for analyzing your own scripts and for obfuscation research. Respect the licenses and rights of other authors' code.

## License

Licensed under the [Apache License 2.0](LICENSE).

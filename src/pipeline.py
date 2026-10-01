import sys
from pathlib import Path

from .ast_nodes import Name
from .graph import build_cfg
from .lift import lift_program
from .clean import clean_program
from .cleanup import cleanup_function
from .constants import decrypt_constants
from .structure import structure_function
from .lexer import LuaLexError
from .parser import LuaParseError, parse_lua
from .strings_layer import _walk, decode_string_layer
from .printer import Printer, render
from .transform import fold_block
from .transitions import EXIT
from .wearedevs import WeAreDevsFormatError, deobfuscate_wearedevs


def _state_text(target, expr, printer):
    if target == EXIT:
        return "exit"
    if target is not None:
        return str(target)
    return printer.render_expr(expr)


def _describe_jump(jump, printer):
    if jump is None:
        return "exit"
    if jump.kind == "exit":
        return "exit"
    if jump.kind == "const":
        return f"goto {jump.target}"
    if jump.kind == "cond":
        cond = printer.render_expr(jump.cond)
        yes = _state_text(jump.true_target, jump.true_expr, printer)
        no = _state_text(jump.false_target, jump.false_expr, printer)
        return f"if {cond} then goto {yes} else goto {no}"
    return f"dynamic {printer.render_expr(jump.raw)}"


def _describe_path(path, printer):
    parts = []
    for cond, sense in path:
        text = printer.render_expr(cond)
        parts.append(text if sense else f"not ({text})")
    return " and ".join(parts)


def build_block_report(cfg):
    printer = Printer()
    counts = {}
    sections = []
    for leaf, node in zip(cfg.dispatcher.leaves, cfg.nodes):
        kind = node.jump.kind if node.jump else "none"
        counts[kind] = counts.get(kind, 0) + 1
        lines = [f"-- block {leaf.index}", f"-- when: {_describe_path(leaf.path, printer)}"]
        for stat in node.body:
            lines.append(printer.render_stat(stat, 0))
        lines.append(f"-- => {_describe_jump(node.jump, printer)}")
        sections.append("\n".join(lines))
    header = [f"-- state variable: {cfg.dispatcher.state_var}", f"-- blocks: {len(cfg.nodes)}"]
    header.extend(f"-- {kind}: {count}" for kind, count in sorted(counts.items()))
    return "\n".join(header) + "\n\n" + "\n\n".join(sections) + "\n"


def build_cfg_report(cfg):
    reachable = set()
    for function in cfg.functions:
        reachable.update(function.nodes)
    lines = [
        f"dispatcher function: {cfg.dispatcher_name}",
        f"closure factories: {', '.join(cfg.factories)}",
        f"functions: {len(cfg.functions)}",
        f"blocks: {len(cfg.nodes)} (reachable from function entries: {len(reachable)})",
        f"unresolved transitions: {sum(1 for n in cfg.nodes if n.unresolved)}",
        f"unmatched states: {len(cfg.unmatched_states)}",
        "",
    ]
    for function in cfg.functions:
        nodes = ", ".join(str(i) for i in function.nodes)
        lines.append(f"function state={function.entry_state} entry_block={function.entry_node} blocks=[{nodes}]")
    lines.append("")
    for node in cfg.nodes:
        targets = [str(t) for t in node.successors]
        if node.exits:
            targets.append("exit")
        if node.unresolved:
            targets.append("?")
        states = ",".join(str(v) for v in sorted(node.entry_states))
        flag = "" if node.index in reachable else " unreferenced"
        lines.append(f"block {node.index} entry_states=[{states}] -> {' '.join(targets)}{flag}")
    return "\n".join(lines) + "\n"


def build_structured_output(cfg):
    printer = Printer()
    parts = ["local ENV = getfenv and getfenv() or _ENV", "local unpack = unpack or table.unpack", ""]
    stats = {"structured": 0, "state_machine": 0, "failed": 0}
    results = []
    for function in cfg.functions:
        try:
            result = structure_function(cfg, function)
        except (RecursionError, ValueError, KeyError, IndexError, AssertionError):
            result = None
        if result is None:
            stats["failed"] += 1
            continue
        stats["structured" if result.structured else "state_machine"] += 1
        results.append(result)

    bodies, decrypt_report = decrypt_constants([r.stats for r in results])
    for result, body in zip(results, bodies):
        result.stats = cleanup_function(body)
    stats["decrypted_strings"] = decrypt_report.decrypted
    stats["rejected"] = decrypt_report.rejected

    for result in results:
        function = result.function
        lines = [f"local function fn_{function.entry_state}({result.header})"]
        if result.locals:
            lines.append(f"    local {', '.join(result.locals)}")
        for stat in result.prologue:
            lines.append(printer.render_stat(stat, 1))
        for stat in result.stats:
            lines.append(printer.render_stat(stat, 1))
        lines.append("end")
        parts.append("\n".join(lines))
        parts.append("")
    lifted_text = None
    try:
        lifted, lift_report = lift_program(cfg, results)
    except (RecursionError, ValueError, KeyError, IndexError, AssertionError):
        lifted, lift_report = None, None
    if lifted is not None:
        names = {item.name for item in _walk(lifted) if isinstance(item, Name)}
        header = []
        if "ENV" in names:
            header.append("local ENV = getfenv and getfenv() or _ENV")
        if "unpack" in names:
            header.append("local unpack = unpack or table.unpack")
        body = "\n".join(printer.render_stat(stat, 0) for stat in lifted)
        lifted_text = "\n".join(header + ([""] if header else []) + [body]) + "\n"
        stats["lifted_closures"] = lift_report.lifted
        stats["unlifted_closures"] = lift_report.unlifted
        stats["cells"] = lift_report.cells
        stats["dead_cells_removed"] = lift_report.removed
        stats["unresolved_slots"] = lift_report.unresolved_slots
    clean_text = None
    if lifted is not None:
        try:
            clean_stats, found = clean_program(lifted)
        except (RecursionError, ValueError, KeyError, IndexError, AssertionError):
            clean_stats, found = None, False
        if clean_stats is not None:
            names = {item.name for item in _walk(clean_stats) if isinstance(item, Name)}
            header = []
            if "ENV" in names:
                header.append("local ENV = getfenv and getfenv() or _ENV")
            if "unpack" in names:
                header.append("local unpack = unpack or table.unpack")
            body = "\n".join(printer.render_stat(stat, 0) for stat in clean_stats)
            clean_text = "\n".join(header + ([""] if header else []) + [body]) + "\n"
            stats["tamper_branch_found"] = found
    return "\n".join(parts) + "\n", stats, lifted_text, clean_text


def run(path: Path, output_dir: Path, with_blocks: bool = True) -> list[Path]:
    source = path.read_text(encoding="utf-8", errors="replace")
    stem = path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(name: str, content: str) -> None:
        target = output_dir / name
        target.write_text(content, encoding="utf-8")
        written.append(target)

    try:
        tree = fold_block(parse_lua(source))
    except (LuaParseError, LuaLexError, RecursionError) as error:
        print(f"{path.name}: parse failed ({error})", file=sys.stderr)
        return written

    report = decode_string_layer(tree)
    if report.found:
        print(
            f"{path.name}: string layer decoded "
            f"(entries={report.decoded_entries}/{report.table_size}, "
            f"resolved={report.resolved_calls}, unresolved={report.unresolved_calls}, "
            f"prologue_removed={report.prologue_removed})",
            file=sys.stderr,
        )
    else:
        try:
            legacy = deobfuscate_wearedevs(source)
            tree = fold_block(parse_lua(legacy))
        except (WeAreDevsFormatError, ValueError, LuaParseError, LuaLexError, RecursionError) as error:
            print(f"{path.name}: string layer not decoded ({error})", file=sys.stderr)

    tree = fold_block(tree)
    folded = tree
    text = render(folded) + "\n"
    save(f"{stem}.decoded.lua", text)
    save(f"{stem}.clean.lua", text)

    if with_blocks:
        try:
            cfg = build_cfg(folded)
        except ValueError as error:
            print(f"{path.name}: dispatcher not found ({error})", file=sys.stderr)
        else:
            save(f"{stem}.blocks.txt", build_block_report(cfg))
            save(f"{stem}.cfg.txt", build_cfg_report(cfg))
            structured_text, structured_stats, lifted_text, clean_text = build_structured_output(cfg)
            save(f"{stem}.structured.lua", structured_text)
            if lifted_text is not None:
                save(f"{stem}.lifted.lua", lifted_text)
            if clean_text is not None:
                save(f"{stem}.script.lua", clean_text)
            print(f"{path.name}: structured functions {structured_stats}", file=sys.stderr)
    return written

from __future__ import annotations

import json
import os
import sys
import time
import types


def _install_source_package_stub() -> None:
    root = os.environ.get("ECSA_DREAMCODER_SOURCE_ROOT")
    if not root:
        raise RuntimeError("ECSA_DREAMCODER_SOURCE_ROOT is required")
    package = types.ModuleType("dreamcoder")
    package.__path__ = [os.path.join(root, "dreamcoder")]
    package.__package__ = "dreamcoder"
    sys.modules["dreamcoder"] = package


_install_source_package_stub()


def _binary(name, fn):
    from dreamcoder.program import Primitive
    from dreamcoder.type import arrow, tbool

    return Primitive(
        name,
        arrow(tbool, tbool, tbool),
        lambda left: lambda right: fn(left, right),
    )


def _unary(name, fn):
    from dreamcoder.program import Primitive
    from dreamcoder.type import arrow, tbool

    return Primitive(name, arrow(tbool, tbool), fn)


def _primitive(name):
    if name == "not":
        return _unary("not", lambda value: not value)
    if name == "and":
        return _binary("and", lambda left, right: left and right)
    if name == "or":
        return _binary("or", lambda left, right: left or right)
    if name == "xor":
        return _binary("xor", lambda left, right: bool(left) ^ bool(right))
    raise ValueError(f"unsupported boolean primitive: {name}")


def _register_primitives(names):
    return [_primitive(name) for name in names]


def _synthesize(payload) -> dict:
    from dreamcoder.grammar import Grammar
    from dreamcoder.task import Task
    from dreamcoder.type import Context, arrow, tbool

    arity = int(payload["arity"])
    primitive_names = tuple(payload["primitives"])
    examples = payload["examples"]
    maximum_mdl = float(payload["maximum_mdl"])
    maximum_programs = int(payload["maximum_programs"])
    timeout_seconds = float(payload["timeout_seconds"])
    evaluation_timeout = float(payload["evaluation_timeout"])

    request = arrow(*([tbool] * arity + [tbool]))
    task = Task(
        "ecsa-program-repair",
        request,
        [
            (tuple(example["inputs"]), example["output"])
            for example in examples
        ],
    )
    grammar = Grammar.uniform(_register_primitives(primitive_names))

    enumerated = 0
    lower = 0.0
    increment = 1.5
    deadline = time.monotonic() + timeout_seconds
    best = None

    while lower < maximum_mdl and enumerated < maximum_programs:
        if time.monotonic() >= deadline:
            break
        upper = min(maximum_mdl + 1e-9, lower + increment)
        shell_hits = []

        for log_prior, _, program in grammar.enumeration(
            Context.EMPTY,
            [],
            request,
            upperBound=upper,
            lowerBound=lower,
            maximumDepth=99,
        ):
            enumerated += 1
            if task.check(program, timeout=evaluation_timeout):
                shell_hits.append(
                    {
                        "program": str(program),
                        "log_prior": float(log_prior),
                        "mdl": float(-log_prior),
                    }
                )
            if enumerated >= maximum_programs or time.monotonic() >= deadline:
                break

        if shell_hits:
            best = min(
                shell_hits,
                key=lambda item: (item["mdl"], item["program"]),
            )
            break
        lower = upper

    result = {
        "status": "found" if best is not None else "not_found",
        "programs_enumerated": enumerated,
        "request": str(request),
    }
    if best is not None:
        result.update(best)
    return result


def _evaluate(payload) -> dict:
    from dreamcoder.program import Program

    primitive_names = tuple(payload["primitives"])
    _register_primitives(primitive_names)
    program = Program.parse(payload["program"])
    value = program.evaluate([])
    for argument in payload["inputs"]:
        value = value(bool(argument))
    if type(value) is not bool:
        raise TypeError(f"DreamCoder program returned non-bool value: {value!r}")
    return {"status": "ok", "output": value}


def main() -> None:
    payload = json.load(sys.stdin)
    mode = payload.get("mode", "synthesize")
    if mode == "synthesize":
        result = _synthesize(payload)
    elif mode == "evaluate":
        result = _evaluate(payload)
    else:
        raise ValueError(f"unsupported worker mode: {mode}")
    json.dump(result, sys.stdout, sort_keys=True)


if __name__ == "__main__":
    main()

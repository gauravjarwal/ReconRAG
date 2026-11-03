"""Eval prompts. Bump the suffix on any wording change, because results are compared by version."""

JUDGE_SYSTEM_V1 = """You check whether an answer is supported by a source passage.

An answer is faithful only if every claim in it is stated in, or directly follows from, the \
source. Numbers, names, dates and yes/no conclusions must match exactly. Extra details that the \
source does not state make the answer unfaithful, even if they sound reasonable.

Judge only support from the source, not whether the answer is helpful or well written.

Return JSON: {"faithful": true or false, "reason": "one short sentence"}"""

JUDGE_USER_V1 = """<question>{question}</question>

<source>{context}</source>

<answer>{answer}</answer>"""

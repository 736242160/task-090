#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chunk_parser.py — 跨块结构解析器（纯 Python 标准库，单文件）

功能
----
输入按任意边界切分的文本块序列，逐块推进解析，输出完整的结构树与错误报告。
结构（括号嵌套、字符串、注释）可以跨越块边界，解析状态在块之间完整保持。

一、状态定义与切换规则（自定，附理由）
------------------------------------
1. NORMAL       结构层。识别括号 { } [ ] ( )、字符串起始引号 " '、注释起始 // 与 /*。
2. STRING       由 " 或 ' 进入。只有与起始引号相同的未转义引号才能关闭；
                反斜杠 \\ 转义紧随其后的任意一个字符。允许字符串跨行、跨块
                （理由：分块传输场景下字符串被切开是常态，强行按行报错会产生
                大量误报；是否允许跨行是语法策略，这里选择宽松策略）。
3. LINE_COMMENT 由 // 进入，遇到换行符 \\n 退出；若块尾无换行则跨块延续，
                输入结束时自然收尾（不算错误，与 C++/Python 行为一致）。
4. BLOCK_COMMENT 由 /* 进入，由 */ 退出，不嵌套（与 C 一致；嵌套注释会显著
                增加误报面，且本工具的目标语言不需要）。

二、优先级规则（同一位置多条规则同时适用时的歧义裁决）
----------------------------------------------------
P1 活跃状态优先（豁免最高优先级）：处于 STRING / *_COMMENT 中时，只有
   “关闭当前状态”的标记与转义符生效，其余一切字符（括号、引号、注释符）
   都是普通字符。理由：词法状态必须具有“粘性”，这与 C/Python/JSON 等
   真实词法器一致，是“字符串、注释里的结构标记不算”的直接实现。
P2 NORMAL 下注释起始符优先于字符串与括号：/ 后紧跟 / 或 * 即进入注释。
   理由：注释可以吞掉包括引号在内的任意字符，若让字符串优先，则
   `// "` 这类注释会被错误地打开一个字符串。
P3 字符串起始符优先于括号：先判引号，最后才判括号。理由：括号是结构层
   最低优先级，只有确认当前字符不属于任何豁免构造时才参与结构配对。
P4 闭合括号与栈顶不匹配：报 mismatched_closer 错误并丢弃该闭合符
   （不弹栈），继续解析。理由：保留栈可让后续正确的闭合符继续配对，
   避免一次笔误引发级联误报。

三、跨块保持的状态
------------------
state（当前状态）、quote（字符串引号种类）、escape（转义挂起）、
star（块注释中上一字符是否为 *）、pending_slash（NORMAL 下块尾的孤立 /，
其位置随字符一起保存，下一块首字符决定它是否开启注释）、括号栈、
当前未闭合的字符串/注释节点。因此以下跨块衔接都能正确处理：
  - 字符串在块中间被切开；块尾 \\ 转义下一块首字符；
  - 块注释在块中间被切开；结束标记 */ 被切成 '*' | '/' 两块；
  - 行注释无换行跨块延续；注释起始符 // 或 /* 被切成两块；
  - 括号嵌套跨任意多个块保持。

四、错误报告
------------
unclosed_string / unclosed_block_comment / unclosed_bracket：
最后一个块结束时仍未闭合，报告其起始位置（chunkN:L行:C列）。
unmatched_closer / mismatched_closer：闭合符无对应或类型不匹配，
报告闭合符位置、期望与实际。

用法
----
    python3 chunk_parser.py                 # 运行内置示例（覆盖全部跨块情形）
    python3 chunk_parser.py chunks.json     # 从文件读取 {"chunks": ["...", ...]}
    python3 chunk_parser.py --json          # 机器可读 JSON 输出
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from enum import Enum


class State(Enum):
    NORMAL = "NORMAL"
    STRING = "STRING"
    LINE_COMMENT = "LINE_COMMENT"
    BLOCK_COMMENT = "BLOCK_COMMENT"


OPENERS = {"{": "}", "[": "]", "(": ")"}
OPENER_KIND = {"{": "brace", "[": "bracket", "(": "paren"}
KIND_OPENER = {v: k for k, v in OPENER_KIND.items()}
CLOSERS = {"}": "{", "]": "[", ")": "("}
QUOTES = {'"', "'"}
LEAF_KINDS = ("string", "line_comment", "block_comment")


@dataclass
class Pos:
    """位置：第几块、块内行号、块内列号（均从 1/0 起，chunk 从 0 起）。"""
    chunk: int
    line: int
    col: int

    def __str__(self) -> str:
        return f"chunk{self.chunk}:L{self.line}:C{self.col}"


@dataclass
class Node:
    kind: str
    start: Pos | None = None
    end: Pos | None = None
    children: list["Node"] = field(default_factory=list)
    text: str = ""
    closed: bool = False

    def to_dict(self) -> dict:
        data: dict = {"kind": self.kind, "closed": self.closed}
        if self.start is not None:
            data["start"] = str(self.start)
        if self.end is not None:
            data["end"] = str(self.end)
        if self.text:
            data["text"] = self.text
        if self.children:
            data["children"] = [c.to_dict() for c in self.children]
        return data


@dataclass
class ParseError:
    code: str
    message: str
    pos: Pos

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "pos": str(self.pos)}


class ChunkParser:
    """逐块喂入文本的增量结构解析器。"""

    def __init__(self) -> None:
        self.state = State.NORMAL
        self.quote = ""
        self.escape = False
        self.star = False
        self.pending_slash_pos: Pos | None = None
        self.root = Node(kind="root", closed=True)
        self.stack: list[Node] = []
        self.errors: list[ParseError] = []
        self.chunk_index = -1
        self._string_node: Node | None = None
        self._comment_node: Node | None = None

    # ---------- 公共 API ----------

    def feed(self, text: str) -> None:
        """喂入一个文本块。块结束时的未完成状态自动保留到下一这块。"""
        self.chunk_index += 1
        line, col = 1, 0
        for ch in text:
            col += 1
            self._step(ch, Pos(self.chunk_index, line, col))
            if ch == "\n":
                line += 1
                col = 0

    def finish(self) -> dict:
        """最后一块喂完后调用：做收尾检查并返回 {tree, errors}。"""
        if self.state is State.STRING and self._string_node is not None:
            node = self._string_node
            self.errors.append(ParseError(
                "unclosed_string",
                f"字符串未闭合（起始引号 {self.quote!r}），起始于 {node.start}",
                node.start))
        elif self.state is State.BLOCK_COMMENT and self._comment_node is not None:
            node = self._comment_node
            self.errors.append(ParseError(
                "unclosed_block_comment",
                f"块注释未闭合（期望 '*/'），起始于 {node.start}",
                node.start))
        elif self.state is State.LINE_COMMENT and self._comment_node is not None:
            self._comment_node.closed = True  # 行注释在输入末尾自然结束

        for node in self.stack:  # 自外向内报告所有未闭合括号
            opener = KIND_OPENER[node.kind]
            self.errors.append(ParseError(
                "unclosed_bracket",
                f"括号 {opener!r} 未闭合，起始于 {node.start}，"
                f"期望闭合符 {OPENERS[opener]!r}",
                node.start))

        return {"tree": self.root.to_dict(),
                "errors": [e.to_dict() for e in self.errors]}

    # ---------- 状态机 ----------

    def _step(self, ch: str, pos: Pos) -> None:
        if self.state is State.STRING:
            self._step_string(ch, pos)
        elif self.state is State.LINE_COMMENT:
            self._step_line_comment(ch, pos)
        elif self.state is State.BLOCK_COMMENT:
            self._step_block_comment(ch, pos)
        else:
            self._step_normal(ch, pos)

    def _step_string(self, ch: str, pos: Pos) -> None:
        node = self._string_node
        node.text += ch
        if self.escape:
            self.escape = False
        elif ch == "\\":
            self.escape = True
        elif ch == self.quote:
            node.end = pos
            node.closed = True
            self._string_node = None
            self.state = State.NORMAL

    def _step_line_comment(self, ch: str, pos: Pos) -> None:
        node = self._comment_node
        node.text += ch
        if ch == "\n":
            node.end = pos
            node.closed = True
            self._comment_node = None
            self.state = State.NORMAL

    def _step_block_comment(self, ch: str, pos: Pos) -> None:
        node = self._comment_node
        node.text += ch
        if self.star and ch == "/":
            node.end = pos
            node.closed = True
            self._comment_node = None
            self.state = State.NORMAL
            self.star = False
        else:
            self.star = (ch == "*")

    def _step_normal(self, ch: str, pos: Pos) -> None:
        # 先结算上一块/上一字符留下的孤立 '/'
        if self.pending_slash_pos is not None:
            start = self.pending_slash_pos
            self.pending_slash_pos = None
            if ch == "/":
                self._open_comment("line_comment", start)
                self.state = State.LINE_COMMENT
                return
            if ch == "*":
                self._open_comment("block_comment", start)
                self.state = State.BLOCK_COMMENT
                self.star = False
                return
            # 孤立的 '/' 不是结构符，按普通字符丢弃，继续处理当前字符

        if ch == "/":
            self.pending_slash_pos = pos
        elif ch in QUOTES:
            node = Node(kind="string", start=pos)
            self._attach(node)
            self._string_node = node
            self.quote = ch
            self.escape = False
            self.state = State.STRING
        elif ch in OPENERS:
            node = Node(kind=OPENER_KIND[ch], start=pos)
            self._attach(node)
            self.stack.append(node)
        elif ch in CLOSERS:
            self._close_bracket(ch, pos)
        # 其余字符为普通内容，忽略

    # ---------- 辅助 ----------

    def _attach(self, node: Node) -> None:
        parent = self.stack[-1] if self.stack else self.root
        parent.children.append(node)

    def _open_comment(self, kind: str, start: Pos) -> None:
        node = Node(kind=kind, start=start)
        self._attach(node)
        self._comment_node = node

    def _close_bracket(self, ch: str, pos: Pos) -> None:
        if not self.stack:
            self.errors.append(ParseError(
                "unmatched_closer",
                f"多余的闭合符 {ch!r}：括号栈为空", pos))
            return
        top = self.stack[-1]
        expected = OPENERS[KIND_OPENER[top.kind]]
        if CLOSERS[ch] == KIND_OPENER[top.kind]:
            top.end = pos
            top.closed = True
            self.stack.pop()
        else:
            # 恢复策略：丢弃该闭合符，不弹栈，继续解析
            self.errors.append(ParseError(
                "mismatched_closer",
                f"闭合符不匹配：得到 {ch!r}，但栈顶是 "
                f"{KIND_OPENER[top.kind]!r}（起始于 {top.start}），"
                f"期望 {expected!r}；已丢弃该闭合符",
                pos))


def parse_chunks(chunks: list[str]) -> dict:
    """便捷 API：喂入全部块并返回 {'tree': ..., 'errors': [...]}。"""
    parser = ChunkParser()
    for chunk in chunks:
        parser.feed(chunk)
    return parser.finish()


# ---------- 输出格式化 ----------

def format_tree(node: Node, depth: int = 0, lines: list[str] | None = None) -> list[str]:
    if lines is None:
        lines = []
    desc = node.kind
    if node.kind in KIND_OPENER:
        desc += f" '{KIND_OPENER[node.kind]}'"
    if node.start is not None:
        desc += f"  @{node.start}"
        desc += f" -> {node.end}" if node.end else " -> <未闭合>"
    if node.kind in LEAF_KINDS:
        preview = node.text.replace("\n", "\\n")
        if len(preview) > 30:
            preview = preview[:30] + "..."
        desc += f"  text={preview!r}"
    if not node.closed and node.kind != "root":
        desc += "  [UNCLOSED]"
    lines.append("  " * depth + desc)
    for child in node.children:
        format_tree(child, depth + 1, lines)
    return lines


def format_report(root: Node, errors: list[ParseError]) -> str:
    out = ["=" * 25 + " 结构树 " + "=" * 25]
    out += format_tree(root)
    out.append("=" * 25 + f" 错误报告（共 {len(errors)} 条） " + "=" * 25)
    if errors:
        for i, err in enumerate(errors, 1):
            out.append(f"{i}. [{err.code}] @{err.pos}  {err.message}")
    else:
        out.append("（无错误）")
    return "\n".join(out)


# ---------- 内置示例 ----------

DEMO_CHUNKS = [
    # chunk0: '{' 入栈；字符串开始；块尾是反斜杠 -> 转义状态跨块
    'root { name: "跨块\\',
    # chunk1: 首字符 '"' 被上一块的 '\\' 转义（仍是字符串内容）；
    #         第二个 '"' 关闭字符串；块尾孤立的 '/' 悬而未决
    '"仍在字符串" /',
    # chunk2: 首字符 '/' 与上一块尾的 '/' 拼成 '//'，行注释开始
    #         （起始位置记在 chunk1）；块尾无换行 -> 注释跨块延续
    '/ 行注释跨块',
    # chunk3: 注释在换行处结束；字符串 "}" 里的 '}' 被豁免；
    #         [ ] 与内层 { } 均在本块内闭合
    '注释继续\n list: [1, {s: "}"}], ',
    # chunk4: 块注释开始，块尾是 '*' -> star 状态跨块
    '/* 块注释 *',
    # chunk5: 首字符 '/' 与上一块尾的 '*' 拼成 '*/'，注释跨块关闭；
    #         '(' 入栈，直到结束也未闭合
    '/ 结束 pair: (a, b',
    # chunk6: ']' 与栈顶 '(' 不匹配 -> mismatched_closer；
    #         字符串开始但永不闭合 -> unclosed_string
    '] tail: "未闭合',
]


def main(argv: list[str]) -> int:
    args = argv[1:]
    as_json = "--json" in args
    files = [a for a in args if not a.startswith("--")]

    if files:
        with open(files[0], encoding="utf-8") as f:
            data = json.load(f)
        chunks = data["chunks"] if isinstance(data, dict) else list(data)
    else:
        chunks = DEMO_CHUNKS
        print("未提供输入文件，运行内置示例。"
              "（用法: python3 chunk_parser.py [chunks.json] [--json]）\n")
        print("输入分块：")
        for i, c in enumerate(chunks):
            print(f"  chunk{i}: {c!r}")
        print()

    parser = ChunkParser()
    for chunk in chunks:
        parser.feed(chunk)
    result = parser.finish()

    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_report(parser.root, parser.errors))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

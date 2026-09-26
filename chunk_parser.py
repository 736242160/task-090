#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chunk_parser.py — 跨块状态保持的增量结构解析器（纯标准库，单文件）

场景：大文件被切成若干文本块顺序喂入，结构（嵌套括号、字符串、注释）
可能跨越块边界。解析器按块推进，块末未完成的状态自动保持到下一块；
最后一块结束时仍未闭合的状态，报告其起始位置。

一、识别的结构（类 C 语法，可自行扩展）
  - 嵌套层级：{} () [] ，输出为结构树
  - 字符串："..." 与 '...'，支持 \\ 转义（含跨块悬挂转义）
  - 注释：// 行注释，/* ... */ 块注释

二、状态机（跨块保持的状态）
  mode: code | string | line_comment | block_comment
  附加状态：字符串引号种类、转义悬挂标志、注释/字符串起始位置、
            块末尾孤立 '/' 的悬挂标志、嵌套栈（含每层的起始位置）

三、歧义消解优先级（高 -> 低），理由：与主流语言词法规则一致，
   "已持有的内层上下文"优先于"新出现的外层标记"：
  1. 字符串内的 \\ 转义：紧跟的字符一律视为字面量（包括引号、换行）
  2. 字符串内：只识别闭合引号，注释标记 // /* 与括号均为内容
  3. 注释内：只识别注释结束符（换行 或 */），引号与括号均为内容
  4. code 态：从左到右扫描，先出现的开符号先生效；
     // 与 /* 优先于单个 / 的判定（块末尾孤立的 / 挂起到下一块判定）

四、错误报告
  - 最后一块结束仍未闭合的字符串 / 块注释 / 嵌套层：报告其起始位置
  - 不匹配的闭合括号（栈顶类型不符或栈空）：报告闭合符位置，忽略该符继续

五、位置表示：(chunk, line, col)，均为 0 基 chunk 序号 + 1 基行列。

用法：
  python3 chunk_parser.py              # 运行内置示例
  python3 chunk_parser.py f1 f2 ...    # 每个文件视为一个块，顺序解析
  或在代码中：from chunk_parser import parse_chunks
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field


# ---------------------------------------------------------------- 数据结构

@dataclass
class Pos:
    chunk: int
    line: int
    col: int

    def __str__(self) -> str:
        return f"chunk{self.chunk}:{self.line}:{self.col}"


@dataclass
class Node:
    """结构树节点。kind: root | brace | paren | bracket"""
    kind: str
    open_pos: Pos | None = None
    close_pos: Pos | None = None
    children: list["Node"] = field(default_factory=list)

    @property
    def closed(self) -> bool:
        return self.close_pos is not None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "open": str(self.open_pos) if self.open_pos else None,
            "close": str(self.close_pos) if self.close_pos else None,
            "closed": self.closed if self.kind != "root" else True,
            "children": [c.to_dict() for c in self.children],
        }


OPENERS = {"{": "brace", "(": "paren", "[": "bracket"}
CLOSERS = {"}": "brace", ")": "paren", "]": "bracket"}


# ---------------------------------------------------------------- 解析器

class ChunkParser:
    def __init__(self) -> None:
        self.mode = "code"               # code|string|line_comment|block_comment
        self.quote: str | None = None    # 字符串引号种类
        self.escaped = False             # 字符串内转义悬挂（可跨块）
        self.string_start: Pos | None = None
        self.comment_start: Pos | None = None
        self.pending_slash = False       # 块末尾孤立 '/'，挂起待下一块判定
        self.slash_pos: Pos | None = None
        self.pending_star = False        # 块注释内块末尾孤立 '*'，挂起判定 */
        self.root = Node("root")
        self.stack: list[Node] = [self.root]
        self.errors: list[dict] = []
        # 当前块内位置
        self.chunk_index = -1
        self.line = 1
        self.col = 1

    # ---- 位置辅助 ----
    def _pos(self) -> Pos:
        return Pos(self.chunk_index, self.line, self.col)

    def _advance(self, ch: str) -> None:
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1

    def _error(self, kind: str, message: str, pos: Pos) -> None:
        self.errors.append({"type": kind, "message": message, "pos": str(pos)})

    # ---- 主入口：喂入一个块 ----
    def feed(self, text: str) -> None:
        self.chunk_index += 1
        self.line = 1
        self.col = 1
        i, n = 0, len(text)

        # 上一块末尾有孤立 '/'：用本块首字符判定它是不是注释起点
        if self.pending_slash:
            if n == 0:
                return  # 空块，继续挂起
            self.pending_slash = False
            first = text[0]
            if first == "/":
                self.mode = "line_comment"
                self.comment_start = self.slash_pos
                self._advance(first)
                i = 1
            elif first == "*":
                self.mode = "block_comment"
                self.comment_start = self.slash_pos
                self._advance(first)
                i = 1
            # 否则只是普通 '/'，忽略

        # 块注释内，上一块末尾有孤立 '*'：本块首字符为 '/' 则闭合注释
        if self.pending_star:
            if n == 0:
                return  # 空块，继续挂起
            self.pending_star = False
            first = text[0]
            if first == "/":
                self.mode = "code"
                self.comment_start = None
                self._advance(first)
                i = 1
            # 否则 '*' 只是注释内容，继续处于 block_comment

        while i < n:
            ch = text[i]
            pos = self._pos()

            if self.mode == "string":
                # 优先级 1：转义符消费下一字符（含引号/换行/反斜杠本身）
                if self.escaped:
                    self.escaped = False
                elif ch == "\\":
                    self.escaped = True
                elif ch == self.quote:
                    self.mode = "code"
                    self.quote = None
                    self.string_start = None
                # 其余字符（含 // /* 括号）均为字符串内容 —— 豁免

            elif self.mode == "line_comment":
                if ch == "\n":
                    self.mode = "code"
                # 其余均为注释内容 —— 豁免

            elif self.mode == "block_comment":
                if ch == "*" and i + 1 < n and text[i + 1] == "/":
                    self._advance(ch)
                    i += 1
                    ch = text[i]
                    self.mode = "code"
                    self.comment_start = None
                elif ch == "*" and i + 1 == n:
                    # 块末尾孤立 '*'：可能是被切开的 */，挂起到下一块判定
                    self.pending_star = True

            else:  # code
                if ch in "\"'":
                    self.mode = "string"
                    self.quote = ch
                    self.string_start = pos
                elif ch == "/":
                    if i + 1 < n and text[i + 1] == "/":
                        self.mode = "line_comment"
                        self.comment_start = pos
                        self._advance(ch)
                        i += 1
                        ch = text[i]
                    elif i + 1 < n and text[i + 1] == "*":
                        self.mode = "block_comment"
                        self.comment_start = pos
                        self._advance(ch)
                        i += 1
                        ch = text[i]
                    elif i + 1 == n:
                        # 块末尾孤立 '/'：可能是被切开的 // 或 /*，挂起
                        self.pending_slash = True
                        self.slash_pos = pos
                    # 否则是普通字符（如除号），忽略
                elif ch in OPENERS:
                    node = Node(OPENERS[ch], open_pos=pos)
                    self.stack[-1].children.append(node)
                    self.stack.append(node)
                elif ch in CLOSERS:
                    kind = CLOSERS[ch]
                    top = self.stack[-1]
                    if top.kind == kind:
                        top.close_pos = pos
                        self.stack.pop()
                    else:
                        expect = {
                            "root": "（无开放括号）",
                        }.get(top.kind, top.kind)
                        self._error(
                            "unmatched_closer",
                            f"闭合符 {ch!r} 与当前开放结构 {expect} 不匹配，已忽略",
                            pos,
                        )
                # 其余普通字符忽略

            self._advance(ch)
            i += 1

    # ---- 最后一块之后调用：结算未闭合状态 ----
    def finish(self) -> None:
        if self.pending_slash:
            self.pending_slash = False  # 文件末尾的孤立 '/'，普通字符
        if self.pending_star:
            self.pending_star = False   # 文件末尾的孤立 '*'，注释内容
        if self.mode == "string":
            self._error(
                "unterminated_string",
                f"字符串未闭合（引号 {self.quote!r}），起始于 {self.string_start}",
                self.string_start,
            )
        elif self.mode == "block_comment":
            self._error(
                "unterminated_comment",
                f"块注释未闭合，起始于 {self.comment_start}",
                self.comment_start,
            )
        # 行注释到文件末尾自然结束，不算错误
        for node in self.stack[1:]:
            self._error(
                "unclosed_block",
                f"嵌套结构 {node.kind} 未闭合，起始于 {node.open_pos}",
                node.open_pos,
            )

    def result(self) -> dict:
        return {"tree": self.root.to_dict(), "errors": self.errors}


def parse_chunks(chunks: list[str]) -> dict:
    """顺序解析若干文本块，返回 {'tree': ..., 'errors': [...]}。"""
    p = ChunkParser()
    for chunk in chunks:
        p.feed(chunk)
    p.finish()
    return p.result()


# ---------------------------------------------------------------- 展示辅助

def print_tree(node: Node, indent: int = 0) -> None:
    names = {"root": "<root>", "brace": "{ }", "paren": "( )", "bracket": "[ ]"}
    pad = "  " * indent
    if node.kind == "root":
        print(f"{pad}{names[node.kind]}")
    else:
        status = f"{node.open_pos} -> {node.close_pos}" if node.closed \
            else f"{node.open_pos} -> <未闭合>"
        print(f"{pad}{names[node.kind]}  {status}")
    for child in node.children:
        print_tree(child, indent + 1)


def show(title: str, chunks: list[str]) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)
    for idx, c in enumerate(chunks):
        print(f"--- chunk {idx} ---")
        print(c if c else "(空块)")
    parser = ChunkParser()
    for chunk in chunks:
        parser.feed(chunk)
    parser.finish()
    print("--- 结构树 ---")
    print_tree(parser.root)
    print("--- 错误清单 ---")
    if parser.errors:
        for e in parser.errors:
            print(f"  [{e['type']}] {e['message']}")
    else:
        print("  （无错误）")
    print()


# ---------------------------------------------------------------- 示例

def demo() -> None:
    # 示例 1：状态跨块衔接，全部正确闭合
    # 覆盖：字符串跨块、字符串内含 } 和 //（豁免）、块注释跨块且内含括号（豁免）、
    #       转义符悬挂在块末尾、嵌套括号跨块
    show("示例 1：跨块衔接（无错误）", [
        'config = {\n  name: "he',
        'llo } // 仍在字符串里",\n  /* 注释 { [ ( ',
        '注释继续 */ data: [1, (2',
        '+3)],\n  path: "a\\',      # 块末尾是转义符，下一字符被转义
        'b"\n}\n',                  # 字符串内容为 a"b，随后 } 闭合 brace
    ])

    # 示例 2：结构错误 —— 不匹配的闭合符 + 未闭合嵌套（各报其起始/出错位置）
    show("示例 2：闭合符不匹配 + 嵌套未闭合", [
        'a = [1, 2\n',
        'b = (\n',
        '} )\n',                   # } 与栈顶 ( 不匹配，被忽略；) 闭合 paren
    ])

    # 示例 3：字符串跨块后仍未闭合 —— 报告字符串起始位置
    show("示例 3：字符串跨块且未闭合", [
        's = "跨块字符串 { [ //\n',  # 内部括号、注释标记全部豁免
        '第二块仍在字符串里',
    ])

    # 示例 4：块注释跨块后仍未闭合 —— 报告注释起始位置
    show("示例 4：块注释跨块且未闭合", [
        '/* 注释开始 { [ (\n',
        '注释继续，括号均被豁免',
    ])

    # 示例 5：块末尾孤立 '/' 被切开成跨块的 //
    show("示例 5：'/' 跨块拼接为行注释", [
        'x = 1 /',                  # 末尾孤立 '/'
        '/ 这是行注释 { (\ny = 2\n',  # 与上一块拼成 //，{ ( 被豁免
    ])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        chunks = []
        for path in sys.argv[1:]:
            with open(path, "r", encoding="utf-8") as f:
                chunks.append(f.read())
        print(json.dumps(parse_chunks(chunks), ensure_ascii=False, indent=2))
    else:
        demo()

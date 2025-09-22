import tree_sitter_language_pack as tslang
from tree_sitter import Parser


# Load languages
LANGUAGES = {
    "python": tslang.get_language("python"),
    "c": tslang.get_language("c"),
    "cpp": tslang.get_language("cpp"),
}


# Language-specific node types for functions and classes
NODE_TYPES = {
    "c": {
        "function": ["function_definition"],
        "class": [],  # C has no classes
    },
    "cpp": {
        "function": ["function_definition", "function_declarator"],
        "class": ["class_specifier", "struct_specifier"],
    },
    "python": {
        "function": ["function_definition"],
        "class": ["class_definition"],
    },
}


def split_code_by_semantics(code: str, language):
    """
    Splits the code into functions and classes based on syntax tree.

    :param code: Source code (str)
    :param language: Language name ("c", "cpp", "python")
    :return: List of extracted code snippets (list[str])
    """

    if language not in LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")

    parser = Parser(LANGUAGES[language])

    code_bytes = code.encode()
    try:
        tree = parser.parse(code_bytes)
    except Exception as error:
        print(error)
        return None

    types = NODE_TYPES[language]

    def walk(node, node_idxs: list):
        for kind, type_names in types.items():
            if node.type in type_names:
                start, end = node.start_byte, node.end_byte
                node_idxs.append((start, end))
        for child in node.children:
            walk(child, node_idxs)

    root_node = tree.root_node
    node_idxs = []
    walk(root_node, node_idxs)
    # Sort the retrieved index pairs by start index
    node_idxs.sort(key=lambda x: x[0])

    # Merge the index pairs using a stack
    stack = []
    split_idxs = []
    for i in range(len(code_bytes)):
        for s, e in node_idxs:
            if i == s:
                if not len(stack): split_idxs.append(i)
                stack.append(i)
            if i == e:
                stack.pop()
                if not len(stack): split_idxs.append(i)

    results = []
    # The final split indexes
    idxs = [0, *split_idxs, len(code_bytes)]
    node_idxs_merge = list(zip(idxs, idxs[1:]))
    for s, e in node_idxs_merge:
        node_code = code_bytes[s:e].decode()
        results.append(node_code.strip())
    # Filter insignificant snippets
    MAX_CHARS = 3
    results = [result for result in results if len(result) >= MAX_CHARS]
    return results

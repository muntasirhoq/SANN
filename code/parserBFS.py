import pandas as pd
import json
from anytree import Node, RenderTree, AsciiStyle, PreOrderIter
import numpy as np
import argparse
import random
import javalang
import os
from config import Config
from anytree.exporter import DotExporter
import re

def get_subtree(root, text_paths, depth, MAX_DEPTH):
    config = Config()
    que = []
    que.append((root, 1))
    while(que):
        node, depth = que.pop(0)
        #print(node.name[1], depth)
        if depth==len(node.name[0]):
            subtree = [node.name[1] for node in PreOrderIter(node, maxlevel=MAX_DEPTH)] #PreOrderIter(node, maxlevel=8)]
            #subtree_string = ' '.join(subtree)
            text_paths.append(subtree)
            for child in node.children:
                que.append((child, depth+MAX_DEPTH))
        #print(subtree)
        else:
            for child in node.children:
                que.append((child, depth))
    
    return text_paths


def get_token(node):
    token = ''
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'  # node.pop()
    elif isinstance(node, javalang.ast.Node):
        token = node.__class__.__name__

    return token

def get_children(root):
    if isinstance(root, javalang.ast.Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item

    return list(expand(children))

def get_trees(current_node, parent_node, order):
    
    token, children = get_token(current_node), get_children(current_node)
    node = Node([order,token], parent=parent_node, order=order)

    for child_order in range(len(children)):
        get_trees(children[child_order], node, order+str(int(child_order)+1))

def extracting_AST(c_code, leaves, MAX_DEPTH):
    """Extracting paths for a given json code.
    Input:
    json_code: json object. The json object of a snap program to be extracted.
    max_length: int. Max length of the path to be restained.
    max_width: int. Max width of the path to be restained.
    hash_path: boolean. if true, MD5 hashed path will be returned to save space.
    hashing_table: Dict. Hashing table for path.
    Return:
    walk_paths: list of AST paths from the json code.
    """
    
    config = Config()
    depth = 1
    
    # Initialize head node of the code.
    head = Node(["1",get_token(c_code)])
    
    # Recursively construct AST tree.
    
    for child_order in range(len(get_children(c_code))):

        get_trees(get_children(c_code)[child_order], head, "1"+str(int(child_order)+1))
        
    #print(RenderTree(head, style=AsciiStyle()).by_attr())
    #DotExporter(head).to_picture("udo.png")
    
    text_paths = []
    
    # leaves unpruned
    if leaves == 1:
        text_paths = get_subtree(head, text_paths, depth, MAX_DEPTH)
    # leaves pruned
    else:
        leaf_nodes = findall_by_attr(head, name="is_leaf", value=True)
        for leaf in leaf_nodes:
            if(leaf.parent is None):
                continue
            p = leaf.parent
            #print(leaf.name)
            children = []
            for child in p.children:
                if child!=leaf:
                    children.append(child)
            p.children = children
        text_paths = get_subtree(head, text_paths, depth, MAX_DEPTH)
    return text_paths

def program_parser(func):
    tokens = javalang.tokenizer.tokenize(func)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    return tree

def parser_main(code_list, Y, MAX_DEPTH):
    config = Config()
    
    leaf_flag = config.leaf_flag # 1: with leaves, else: pruned
    
    print("Parsing Start")
    
    parsed_code = []
    
    for c in code_list:
        try:
            parsed = program_parser(c)
        except:
            parsed = "Uncompilable"
        parsed_code.append(parsed)
        
    print("----Parsing finished. Extracting subtrees----")
    
    # Extracting paths for all programs in the csv file. Output is [["start,path_hash/path,end",...,...],...].
    raw_paths = [extracting_AST(c_code, leaf_flag, MAX_DEPTH) for c_code in parsed_code]
    print("----Extracting subtrees finished----")
    #print(raw_paths)
    
    concatanated = []
    for problem in raw_paths:
        problem_con = []
        for subtree in problem:
            subtree = list(filter(None, subtree))
            subtree = '$$@$$'.join(subtree)
            problem_con.append(subtree)
        problem_con = '@@$@@'.join(problem_con)
        concatanated.append(problem_con)
        
    main_df = pd.DataFrame(list(
        zip(code_list, concatanated, Y)),
               columns =['Code', 'raw_ast', 'Score'])
    
    main_df = main_df[main_df["raw_ast"] != "Uncompilable"]
    main_df = main_df[main_df["raw_ast"] != "n"]
    
    return main_df
        
    """
    ast = parsed_code[0]
    parsed_code[0].show(showcoord=False)
    print(ast.__class__.__name__)
    
    for name, child in ast.children():
        print(child.__class__.__name__)
    #print(ast.children()[0][1])
    """
#!/usr/bin/env python
# Tree utils for completing brackets and filling out missing words 
# Update 11/16: also add functions for computing precision, recall, f1
# based on berkeley parse analyzer by Jonathan Kummerfeld 
# https://github.com/jkkummerfeld/berkeley-parser-analyser/

import os
import sys
import argparse
import random
import re
#from nlp_util import pstree, render_tree, nlp_eval, treebanks, relaxed_parse_errors


def merge_dels(token_list):
    new_list = []
    for i, s in enumerate(token_list):
        current_s = s
        prev_s = token_list[i-1] if i>0 else None
        if prev_s == "TO_DELETE" and current_s == "TO_DELETE": continue
        else: new_list.append(current_s)
    return new_list

def add_brackets(toks):
    line = ' '.join(toks)
    num_open = line.count('(')
    num_close = line.count(')')
    if num_open == num_close:
        full_sent = toks[:]
        valid = 1
    else:
        valid = 0
        if num_open < num_close:
            add_open = num_close - num_open
            extra_open = ['(']*add_open
            full_sent = extra_open + toks
        else:
            add_close = num_open - num_close
            extra_close = [')']*add_close
            full_sent = toks + extra_close
    return full_sent, valid

def match_length(parse, sent):
    line = ' '.join(parse)
    PUNC = ['.', ',', ':', '``', '\'\'', ';', '?', '!', '$', '"', '%', '*', '&']
    tree = []
    sent_toks = sent[:]
    dec_toks = parse[:]
    num_toks = len(sent_toks)
    num_parse = line.count('XX') 
    num_puncs = sum([line.count(x) for x in PUNC])
    num_out = num_puncs + num_parse
    if num_toks == num_out:
        new_tree = dec_toks[:]
    else:
        if num_out < num_toks: # add 'XX' in this case
            num_X = num_toks - num_out  
            for _ in range(num_X):
                if len(dec_toks) > 3:
                    x_add = random.choice(range(len(dec_toks) - 2)) 
                    # offset a bit so never insert at very beginning or very end
                    dec_toks.insert(x_add + 2, 'XX')
                else:
                    dec_toks.insert(1, 'XX')
            new_tree = dec_toks[:]
        else: # remove XXs 
            num_X = num_out - num_toks
            x_indices = [i for i, x in enumerate(dec_toks) if x == "XX"]
            if num_X < len(x_indices):
                x_remove = random.sample(set(x_indices), num_X)
                for k in x_remove:
                    dec_toks[k] = "TO_DELETE"
                for _ in range(len(x_remove)):
                    dec_toks.remove("TO_DELETE")
            # else: do nothing
            new_tree = dec_toks[:]
    return new_tree

def delete_empty_constituents(parse):
    new_tree = parse[:]
    for i in range(len(new_tree)-1):
        this_tok = new_tree[i]
        next_tok = new_tree[i+1]
        if this_tok[0] == '(' and next_tok[0] == ')':
            new_tree[i] = "TO_DELETE"
            new_tree[i+1] = "TO_DELETE"

    num_del = new_tree.count("TO_DELETE")
    for _ in range(num_del): 
        new_tree.remove("TO_DELETE")
    if num_del == 0:
        return new_tree
    else:
        return delete_empty_constituents(new_tree)

# old version of delete_empty_constituents
# don't use
def delete_empty_constituents_2(parse):
    new_tree = parse[:]
    for i in range(len(new_tree)-1):
        this_tok = new_tree[i]
        next_tok = new_tree[i+1]
        if this_tok[0] == '(' and next_tok[0] == ')':
            new_tree[i] = "TO_DELETE"
            new_tree[i+1] = "TO_DELETE"

    # There are a few cases of nested empty constituents,
    # take care of such cases:
    # merge consecutive "TO_DELETE" tokens
    tok_tmp = merge_dels(new_tree)
    del_constituents = [i for i, x in enumerate(tok_tmp) if 
            x == "TO_DELETE" and (i+1) < len(tok_tmp) and tok_tmp[i+1][0] == ")" and 
            tok_tmp[i-1][0] =="("]
    while len(del_constituents) > 0:
        for idx in del_constituents:
            if tok_tmp[idx+1] == ")" or tok_tmp[idx+1][:2] == ")_":
                tok_tmp[idx-1:idx+2] = ["TO_DELETE"]*3
            else: 
                # this is to take care of the difference between single ')'
                # or things like '))))'
                tok_tmp[idx-1:idx+1] = ["TO_DELETE"]*2
                tok_tmp[idx+1] = tok_tmp[idx+1][1:]
            tok_tmp = merge_dels(tok_tmp)
            del_constituents = [i for i, x in enumerate(tok_tmp)
                    if x == "TO_DELETE" and (i+1) < len(tok_tmp) and tok_tmp[i+1][0] == ")" and
                    tok_tmp[i-1][0] == "("]
    
    num_del = tok_tmp.count("TO_DELETE")
    for _ in range(num_del): 
        tok_tmp.remove("TO_DELETE")
    return tok_tmp


def merge_sent_tree(parse, sent):
    tree = []
    word_idx = 0
    for token in parse:
        tok = token
        if token == 'XX': 
            if word_idx < len(sent):
                tok = '(XX {})'.format(sent[word_idx])
            else:
                tok = '(. .)'
                #sys.stderr.write('Warning: less XX than word!\n')
            word_idx += 1
        elif token[0] == ')':
            tok = ')'
        elif token[0] != '(':
            if word_idx < len(sent):
                tok = '({} {})'.format(token, sent[word_idx])
            else:
                tok = '(. .)'
                #sys.stderr.write('Warning: less XX than word!\n')
            word_idx += 1
        tree.append(tok)
    new_tree = []
    idx = 0
    k = 0
    while idx < len(tree):
        token = tree[idx]
        if token == ')':
            k = 1
            while (idx + k) < len(tree):
                if tree[idx+k] != ')':
                    break
                k += 1
            token = ')' * k
            idx += k - 1
        idx += 1
        new_tree.append(token)

    return new_tree

'''
def compute_overall_score(gold_file, test_file):
    gold_in = open(gold_file).readlines()
    test_in = open(test_file).readlines()
    stats = {'out_evalb': [0, 0, 0],
            'out_relaxed': [0, 0, 0]
            }

    assert len(gold_in) == len(test_in)
    
    for i in range(len(gold_in)):
        print "Sent: " + str(i)
        gold_text = gold_in[i]
        test_text = test_in[i]
        if gold_text == '' and test_text == '':
            break
        elif gold_text == '':
            break
        elif test_text == '':
            break

        gold_text = gold_text.strip()
        test_text = test_text.strip()
        if len(gold_text) == 0:
            continue
        elif len(test_text) == 0:
            continue

        gold_complete_tree = pstree.tree_from_text(gold_text, allow_empty_labels=True)
        gold_complete_tree = treebanks.homogenise_tree(gold_complete_tree)
        treebanks.ptb_cleaning(gold_complete_tree)
        gold_tree = gold_complete_tree
        #gold_tree = treebanks.apply_collins_rules(gold_complete_tree, False)

        test_complete_tree = pstree.tree_from_text(test_text, allow_empty_labels=True)
        test_complete_tree = treebanks.homogenise_tree(test_complete_tree)
        treebanks.ptb_cleaning(test_complete_tree)
        test_tree = test_complete_tree
        #test_tree = treebanks.apply_collins_rules(test_complete_tree, False)
        
        gold_words = gold_tree.word_yield()
        test_words = test_tree.word_yield()
        if len(test_words.split()) != len(gold_words.split()):
            print "Sentence lengths do not match in sentence..." + str(i)
            print "Gold: " + gold_words.__repr__()
            print "Test: " + test_words.__repr__()

        match_strict, gold_strict, test_strict, _, _ = relaxed_parse_errors.counts_for_prf(
                test_tree, gold_tree)
        match_relaxed, gold_relaxed, test_relaxed , _, _ = relaxed_parse_errors.relaxed_counts_for_prf(
                test_tree, gold_tree)
        stats['out_evalb'][0] += match_strict
        stats['out_evalb'][1] += gold_strict
        stats['out_evalb'][2] += test_strict
        p, r, f = nlp_eval.calc_prf(match_strict, gold_strict, test_strict)
        print "Eval--Strict Evalb: %.2f  %.2f  %.2f" % (p*100, r*100, f*100)

        stats['out_relaxed'][0] += match_relaxed
        stats['out_relaxed'][1] += gold_relaxed
        stats['out_relaxed'][2] += test_relaxed
        p, r, f = nlp_eval.calc_prf(match_relaxed, gold_relaxed, test_relaxed)
        print "Eval--Relaxed Edit: %.2f  %.2f  %.2f" % (p*100, r*100, f*100) 

    match = stats['out_evalb'][0]
    gold = stats['out_evalb'][1]
    test = stats['out_evalb'][2]
    p, r, f = nlp_eval.calc_prf(match, gold, test)
    print "Overall--Standard EVALB %s: %.2f  %.2f  %.2f" % ('out', p*100, r*100, f*100)

    match = stats['out_relaxed'][0]
    gold = stats['out_relaxed'][1]
    test = stats['out_relaxed'][2]
    p, r, f = nlp_eval.calc_prf(match, gold, test)
    print "Overall--Relaxed EDIT %s: %.2f  %.2f  %.2f" % ('out', p*100, r*100, f*100)

if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='Test tree_utils functions')
    pa.add_argument('--gf', help='gold file')
    pa.add_argument('--tf', help='test file')
    args = pa.parse_args()
    gold_file = args.gf
    test_file = args.tf
    compute_overall_score(gold_file, test_file)

'''



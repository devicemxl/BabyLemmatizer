#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from argparse import ArgumentParser
import preprocessing
import conlluplus

"""===========================================================
Rawtext -> CoNLL-U+ for BabyLemmatizer 2

asahala 2023
https://github.com/asahala

University of Helsinki
   Origins of Emesal Project
   Centre of Excellence for Ancient Near-Eastern Empires

==========================================================="""

def normalize(xlit):
    xlit = xlit.replace('sz', 'š')
    xlit = xlit.replace('SZ', 'Š')
    xlit = xlit.replace('s,', 'ṣ')
    xlit = xlit.replace('t,', 'ṭ')
    xlit = preprocessing.lowercase_determinatives(xlit)
    xlit = preprocessing.unify_h(xlit)
    xlit = preprocessing.subscribe_indices(xlit)
    return xlit


def upl_to_conllu(upl_file, output):
    """ Convert unit-per-line format into CoNLL-U (archivo → archivo)

    :param upl_file            upl file name
    :param output              CoNLL-U file name

    Example of the input format (line-by-line):

    šum-ma a-wi-lum
    in DUMU a-wi-lim uh₂-ta-ap-pi-id
    in-šu u-hap-pa-du
 
    """
    head = {1: '0'}
    deprel = {1: 'root'}

    with open(upl_file, 'r', encoding='utf-8') as f, \
         open(output, 'w', encoding='utf-8') as o:

        for line in f.read().splitlines():
            i = 1
            if line.startswith('#'):
                o.write(line + '\n')
                continue
            for word in line.strip().split(' '):
                hh = head.get(i, '1')
                rr = deprel.get(i, 'child')
                o.write(f'{i}\t{normalize(word)}\t_\t_\t_\t_\t{hh}\t{rr}\t_\t_\n')
                i += 1
            o.write('\n')

    print(f'> File converted to CoNLL-U+ and saved as {output}')


def txt_lines_to_conllu(lines, output=None):
    """
    Convert list[str] (unit-per-line) into ConlluPlus object or file.

    :param lines: list[str] - cada string es una línea textual
    :param output: str (opcional) - si se proporciona, guarda archivo .conllu
    
    :return: conlluplus.ConlluPlus object (siempre)
    
    Usage:
        # Como librería (solo objeto en memoria):
        conllu_obj = txt_lines_to_conllu(["e-nu-ma e-liš", "šap-liš am-ma-tum"])
        
        # Como híbrido (objeto + archivo):
        conllu_obj = txt_lines_to_conllu(lines, output="salida.conllu")
    """
    head = {1: '0'}
    deprel = {1: 'root'}

    data = []
    comments = []
    sentence = []

    for line in lines:
        line = line.strip()

        if not line:
            # sentence boundary
            if sentence:
                data.append((comments, sentence))
                comments = []
                sentence = []
            continue

        if line.startswith('#'):
            comments.append(line)
            continue

        words = line.split(' ')
        for i, word in enumerate(words, start=1):
            hh = head.get(i, '1')
            rr = deprel.get(i, 'child')

            fields = [
                str(i),                    # ID
                normalize(word),           # FORM
                '_',                       # LEMMA
                '_',                       # UPOS
                '_',                       # XPOS
                '_',                       # FEATS
                hh,                        # HEAD
                rr,                        # DEPREL
                '_',                       # DEPS
                '_',                       # MISC
                '_',                       # ENG
                '_',                       # NORM
                '_',                       # LANG
                '_',                       # FORMCTX
                '_',                       # XPOSCTX
                '_',                       # SCORE
                '_'                        # LOCK
            ]

            sentence.append(fields)

    # flush last sentence if needed
    if sentence:
        data.append((comments, sentence))

    # Crear ConlluPlus object sin leer archivo
    conllu = conlluplus.ConlluPlus.__new__(conlluplus.ConlluPlus)
    conllu.data = data
    conllu.validate = False
    conllu.word_count = sum(len(sent) for _, sent in data)

    # Si se proporciona output, guardar archivo
    if output:
        with open(output, "w", encoding="utf-8") as o:
            for comments, sentence in data:
                for comment in comments:
                    o.write(comment + '\n')
                for fields in sentence:
                    o.write('\t'.join(fields) + '\n')
                o.write('\n')
        print(f'> File converted to CoNLL-U+ and saved as {output}')

    return conllu


# ==============================
#                        MAIN
# ==============================

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument('--filename', type=str)
    args = ap.parse_args()

    if args.filename:
        txt = args.filename
        fn, ext = os.path.splitext(args.filename)
        conllu = fn + '.conllu'

        upl_to_conllu(txt, conllu)
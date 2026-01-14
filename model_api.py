import os
import sys
import subprocess
from preferences import Context
import preprocessing as PP

""" ===========================================================
API for calling OpenNMT and performing intermediate steps for
BabyLemmatizer 2

Modified to use portable OpenNMT paths instead of hardcoded ones

asahala 2023
https://github.com/asahala

University of Helsinki
   Origins of Emesal Project
   Centre of Excellence for Ancient Near-Eastern Empires

=========================================================== """


def run_tagger(input_file, model_name, output_file, cpu=False):
    """
    Run OpenNMT tagger using portable Python paths
    
    :param input_file: source file path
    :param model_name: model path
    :param output_file: output file path
    :param cpu: use CPU instead of GPU
    """
    
    # Construcción portable del comando
    cmd = [
        sys.executable,  # Python actual del sistema
        "-m", "onmt.bin.translate",
        "-model", model_name,
        "-src", input_file,
        "-output", output_file,
        "-min_length", "1"
    ]
    
    if cpu:
        cmd.extend(["-gpu", "-1"])
    else:
        cmd.extend(["-gpu", "0"])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        # Solo mostrar errores si falló
        if result.returncode != 0:
            print(f"Tagger warning/error (code {result.returncode}):")
            if result.stderr:
                print(result.stderr[:500])
                
    except Exception as e:
        print(f"Error running tagger: {e}")
        raise


def run_lemmatizer(input_file, model_name, output_file, cpu=False):
    """
    Run OpenNMT lemmatizer using portable Python paths
    
    :param input_file: source file path
    :param model_name: model path
    :param output_file: output file path
    :param cpu: use CPU instead of GPU
    """
    
    # Construcción portable del comando
    cmd = [
        sys.executable,  # Python actual del sistema
        "-m", "onmt.bin.translate",
        "-model", model_name,
        "-src", input_file,
        "-output", output_file,
        "-min_length", "1"
    ]
    
    if cpu:
        cmd.extend(["-gpu", "-1"])
    else:
        cmd.extend(["-gpu", "0"])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        # Solo mostrar errores si falló
        if result.returncode != 0:
            print(f"Lemmatizer warning/error (code {result.returncode}):")
            if result.stderr:
                print(result.stderr[:500])
                
    except Exception as e:
        print(f"Error running lemmatizer: {e}")
        raise


def read_results(filename):
    """ Read OpenNMT output file """
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.replace(' ', '').rstrip()


def merge_tags(neural_net_output, conllu_object, output_file, field, fieldctx):
    """ Merge neural net output with the CoNLL-U+ object and generate
    data for the next step in pipeline 

    :param neural_net_output     Path/filename to previous step's output
    :param conllu_object         File for storing the annotations
    :param output_file           Output for next step's input
    :param field                 Which field to populate with the output
    :param fieldctx              Which context field to update

    :type neural_net_output      path/file as str
    :type conllu_object          ConlluPlus obj
    :type output_file            path/file as str or None
    :type field                  str
    :type fieldctx               str or None """

    ## TODO: tee tää suoraan tagger/lemmatisaattorikutsun
    ## yhteydessä, etenkin jos tästä tulee modulaarisempi
    
    annotations = read_results(neural_net_output)
    conllu_object.update_value(field, annotations)

    contexts = {'xpos': Context.lemmatizer_context,
                'form': Context.tagger_context}
    
    if fieldctx is not None:
        ctx_annotations = conllu_object.get_contexts(field, size=contexts[field])
        conllu_object.update_value(fieldctx, ctx_annotations)

    if output_file is not None and fieldctx is not None:
        with open(output_file, 'w', encoding='utf-8') as o_file:
            for form, xposctx in conllu_object.get_contents('form', 'xposctx'):
                o_file.write(PP.make_lem_src(form, xposctx) + '\n')

            
def ___merge_tags(tagged_file, lemma_input, output_file):
    """ This function merges Tagger output with lemmatizer test data
    to create input for lemmatizer evaluation """

    ## TODO: DO THIS IN CONLLUPLUS:: OBSOLETE NOW
    
    def filter_pos(string):
        xlit = string.split(' PREV')[0]
        return xlit
    
    with open(tagged_file, 'r', encoding='utf-8') as t_file,\
         open(lemma_input, 'r', encoding='utf-8') as l_file,\
         open(output_file, 'w', encoding='utf-8') as o_file:

        lemmas = (filter_pos(x) for x in l_file.read().splitlines())
        tag_segments = []
        stack = [ct.EOU[0]]
        for tag in t_file.read().splitlines():
            stack.append(tag)
            if len(stack) == 3:
                tag_segments.append('PREV={} UPOS={} NEXT={}'.format(*stack))
                stack.pop(0)

        tag_segments.append(
            'PREV={} UPOS={} NEXT={}'.format(*stack, ct.EOU[0]))
        
        for lemma, pos in zip(lemmas, tag_segments):
            if lemma == ct.EOU[0]:
                o_file.write(lemma + '\n')
            else:
                o_file.write(f'{lemma} {pos}\n')


def file_to_set(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return set(x.split('\t')[0] for x in f.read().splitlines())
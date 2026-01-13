#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import shutil
import tempfile
import conlluplus
import preprocessing as pp
import model_api
from preferences import Paths, Tokenizer, Context
import postprocess

info = """===========================================================
Lemmatizer pipeline for BabyLemmatizer 2

asahala 2023
https://github.com/asahala

University of Helsinki
   Origins of Emesal Project
   Centre of Excellence for Ancient Near-Eastern Empires

==========================================================="""

def io(message):
    print(f'> {message}')


class Lemmatizer:

    def __init__(self, input_file, fast=False, ignore_numbers=True, output_file=None):
        """
        :param input_file: puede ser:
            - str: ruta a archivo .conllu (modo CLI clásico)
            - conlluplus.ConlluPlus: objeto en memoria (modo librería)
        :param fast: bool
        :param ignore_numbers: bool
        :param output_file: str (opcional) - archivo de salida final
                           Si es None y input_file es objeto, NO escribe archivos
        """
        
        self.ignore_numbers = ignore_numbers
        self.fast = fast
        self.output_file = output_file
        
        # --------------------------------------------------
        # Modo LIBRERÍA: recibe objeto ConlluPlus
        # --------------------------------------------------
        if isinstance(input_file, conlluplus.ConlluPlus):
            self.source_file = input_file
            self.input_file = None
            self.is_memory_mode = True
            
            # Crear directorio temporal para archivos intermedios
            self.temp_dir = tempfile.mkdtemp(prefix='babylem_')
            self.input_path = self.temp_dir
            
            # Generar nombres de archivos intermedios en temp
            fn = os.path.join(self.temp_dir, 'temp')
            self.backup_file = os.path.join(self.temp_dir, 'backup_temp.conllu')
            self.word_forms = fn + '.forms'
            self.tagger_input = fn + '.tag_src'
            self.tagger_output = fn + '.tag_pred'
            self.lemmatizer_input = fn + '.lem_src'
            self.lemmatizer_output = fn + '.lem_pred'
            self.final_output = fn + '.final'
            
            self.line_count = 0
            self.segment_count = 0
            
            return
        
        # --------------------------------------------------
        # Modo CLÁSICO: recibe ruta de archivo
        # --------------------------------------------------
        self.is_memory_mode = False
        path, file_ = os.path.split(input_file)
        f, e = file_.split('.')

        # Path for saving intermediate files
        step_path = os.path.join(path, 'steps')

        try:
            os.mkdir(step_path)
        except FileExistsError:
            pass

        fn = os.path.join(step_path, f)
        self.backup_file = os.path.join(path, f'backup_{f}.conllu')
        self.input_file = input_file
        self.input_path = path
        self.word_forms = fn + '.forms'
        self.tagger_input = fn + '.tag_src'
        self.tagger_output = fn + '.tag_pred'
        self.lemmatizer_input = fn + '.lem_src'
        self.lemmatizer_output = fn + '.lem_pred'
        self.final_output = fn + '.final'
        self.line_count = 0
        self.segment_count = 0
        
        # Load and normalize source CoNLL-U+ file
        self.source_file = conlluplus.ConlluPlus(input_file, validate=False)

        
    def preprocess_source(self):
        self.source_file.normalize()
        formctx = self.source_file.get_contexts('form', size=Context.tagger_context)
        self.source_file.update_value('formctx', formctx)
        
        with open(self.tagger_input, 'w', encoding='utf-8') as pos_src, \
             open(self.word_forms, 'w', encoding='utf-8') as wf:
            
            if not self.is_memory_mode:
                io(f'Generating input data for neural net {self.input_file}')
            else:
                io(f'Generating input data for neural net (memory mode)')
                
            for id_, form, formctx in self.source_file.get_contents('id', 'form', 'formctx'):
                pos_src.write(
                    pp.make_tagger_src(formctx, context=Context.tagger_context) + '\n')
                wf.write(pp.get_chars(form + '\n'))
                self.line_count += 1
                if id_ == '1':
                    self.segment_count += 1
                    
            io(f'Input file size: {self.line_count} words in {self.segment_count} segments.')


    def update_model(self, model_name):
        overrides = [os.path.join(self.input_path, f) for f
                     in os.listdir(self.input_path) if f.endswith('.tsv')]
                
        if overrides:
            mod_o = os.path.join(
                Paths.models, model_name, 'override', 'override.conllu')
            override = conlluplus.ConlluPlus(mod_o, validate=False)
            for o_file in overrides:
                override.read_corrections(o_file)
                override.normalize()
                os.remove(o_file)
            override.write_file(mod_o)

            
    def run_model(self, model_name, cpu):
        """
        Ejecuta el modelo de lemmatización.
        
        :param model_name: nombre del modelo
        :param cpu: bool - usar CPU en lugar de GPU
        :return: conlluplus.ConlluPlus - objeto procesado (siempre)
        """

        # Read Tokenizer Preferences
        Tokenizer.read(model_name)
        Context.read(model_name)
        
        # Update model override
        self.update_model(model_name)

        # En modo clásico, recargar desde archivo
        if not self.is_memory_mode:
            self.source_file = conlluplus.ConlluPlus(
                self.input_file, validate=False)
        
        # Backup for write-protected fields
        if not self.is_memory_mode:
            pp_file = self.input_file.replace('.conllu', '_pp.conllu')
            if os.path.isfile(pp_file):
                is_backup = True
                shutil.copy(pp_file, self.backup_file)
            else:
                is_backup = False
        else:
            is_backup = False
            
        # Preprocess data for lemmatization
        self.preprocess_source()
                
        # Set model paths
        tagger_path = os.path.join(
                Paths.models, model_name, 'tagger', 'model.pt')
        lemmatizer_path = os.path.join(
                Paths.models, model_name, 'lemmatizer', 'model.pt')

        # Run tagger on input
        io(f'Tagging with {model_name}')
        model_api.run_tagger(self.tagger_input,
                             tagger_path,
                             self.tagger_output,
                             cpu)

        # Merge tags to make lemmatizer input
        model_api.merge_tags(self.tagger_output,
                             self.source_file,
                             self.lemmatizer_input,
                             'xpos',
                             'xposctx')

        # Run lemmatizer
        io(f'Lemmatizing with {model_name}')
        model_api.run_lemmatizer(self.lemmatizer_input,
                                 lemmatizer_path,
                                 self.lemmatizer_output,
                                 cpu)

        # Merge lemmata to CoNLL-U+
        model_api.merge_tags(self.lemmatizer_output,
                             self.source_file,
                             None,
                             'lemma',
                             None)

        # En modo clásico, escribir archivo _nn.conllu
        if not self.is_memory_mode:
            self.source_file.write_file(
                self.input_file.replace('.conllu', '_nn.conllu'))

        # Initialize postprocessor
        P = postprocess.Postprocessor(
            predictions=self.source_file,
            model_name=model_name)

        P.initialize_scores()
        P.fill_unambiguous(threshold=0.6)
        P.disambiguate_by_pos_context(threshold=0.6)
        P.apply_override()
        
        if self.ignore_numbers:
            self.source_file.unlemmatize(numbers=True)

        # Temporary field cleanup
        self.source_file.force_value('xposctx', '_')
        self.source_file.force_value('formctx', '_')
        
        # Escribir archivo _pp.conllu solo si:
        # 1. Modo clásico, O
        # 2. Modo memoria pero se especificó output_file
        if not self.is_memory_mode:
            pp_file = self.input_file.replace('.conllu', '_pp.conllu')
            self.source_file.write_file(pp_file, add_info=True)
            
            # Merge backup
            print('> Merging manual corrections')
            
            # Write lemmalists
            self.source_file.make_lemmalists()
            
        elif self.output_file:
            # Modo memoria pero usuario quiere archivo de salida
            self.source_file.write_file(self.output_file, add_info=True)
            print(f'> Output saved to {self.output_file}')

        # Limpiar directorio temporal en modo memoria
        if self.is_memory_mode:
            try:
                shutil.rmtree(self.temp_dir)
            except:
                pass

        # Siempre retornar el objeto procesado
        return self.source_file

        
    def override_cycle(self):
        """ Lemmatization cycle """
        filename, ext = os.path.splitext(self.filename)


if __name__ == "__main__":
    """ Demo for lemmatization """
    pass
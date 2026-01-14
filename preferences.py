import os
import sys

""" BabyLemmatizer 2 Preferences ==================================

  asahala 2023-2024
  github.com/asahala/BabyLemmatizer

  Modified to use portable paths instead of hardcoded ones

=============================================================== """

version_history =\
    "1.0    2022-05-01    TurkuNLP dependent version.\n"\
    "2.0    2023-03-08    Moved to OpenNMT from TurkuNLP.\n"\
    "2.1    2023-09-05    Model versioning --tokenizer.\n"\
    "2.2    2024-06-07    Adjustable context windows."\
    "2.3    2026-01-13    not more hard coded paths. "

__version__ = '2.2'

# ============================================================================
# PORTABLE PATHS - Ya no se usan rutas hardcodeadas
# ============================================================================
# En lugar de rutas hardcodeadas, usamos el Python actual del sistema
# OpenNMT-py debe estar instalado en el entorno actual:
#   pip install OpenNMT-py==3.2.0
#
# Las funciones en model_api.py ahora usan:
#   sys.executable -m onmt.bin.translate
# ============================================================================

# Estas variables se mantienen por compatibilidad pero ya NO se usan
python_path = ''  # DEPRECATED - ahora se usa sys.executable
onmt_path = ''    # DEPRECATED - ahora se usa -m onmt.bin.translate


class Paths:

    """ Container for crucial paths """
    conllu = 'conllu'
    models = 'models'
    override = 'override'

    
class Context:
    
    """ How many word forms are taken into account in POS-tagging """
    tagger_context = 2 # default 2

    """ How many POS-tags are taken into account in lemmatization """
    lemmatizer_context = 1 # default 1

    def read(prefix):
        if not os.path.isfile(os.path.join(Paths.models, prefix, 'config.yaml')):
            print('> Your model was trained with an old version of BabyLemmatizer.')
            print('> Using default contexts')
        else:
            with open(os.path.join(Paths.models, prefix, 'config.yaml')) as f:
                for l in f.read().splitlines():
                    l = l.replace(' ', '')
                    if l.startswith('tagger_context'):
                        val = int(l.split(':')[-1])
                        Context.tagger_context = val
                    elif l.startswith('lemmatizer_context'):
                        val = int(l.split(':')[-1])
                        Context.lemmatizer_context = val

        print(f'> Tagger context = {Context.tagger_context}')
        print(f'> Lemmatizer context = {Context.lemmatizer_context}')

        
class Tokenizer:

    """ This class controls tokenizer behavior 

    0 = Logo-syllabic (Akkadian, Urartian, Hittite, Elamite)
    1 = Sumerian
    2 = Character sequence (Greek, Latin, Persian, Ugaritic etc.)  

    This info is saved in to model config.txt"""
    
    setting = 0

    def read(prefix):
        if not os.path.isfile(os.path.join(Paths.models, prefix, 'config.yaml')):
            print('> Your model was trained with an old version of BabyLemmatizer.')
            print('> Using Tokenizer setting 0. Rebuild model using --tokenizer.')
            Tokenizer.setting = 0
        else:
            with open(os.path.join(Paths.models, prefix, 'config.yaml')) as f:
                for l in f.read().splitlines():
                    l = l.replace(' ', '')
                    if l.startswith('tokenizer'):
                        val = int(l.split(':')[-1])
                        Tokenizer.setting = val
                        
        print(f'> Using tokenizer {Tokenizer.setting}')
        

    
if __name__ == "__main__":
    # Versión portable del test
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, "-m", "onmt.bin.train", "-h"],
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except Exception as e:
        print(f"Error testing OpenNMT: {e}")
        print("\nMake sure OpenNMT-py is installed:")
        print("  pip install OpenNMT-py==3.2.0")
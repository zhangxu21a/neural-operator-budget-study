import importlib.util,pathlib
ROOT=pathlib.Path(__file__).resolve().parent
VENDOR=ROOT/'vendor/mlmc_optim'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,VENDOR/path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
FNO2d=load('author_fno','examples/examples_src/models/fno.py').FNO2d
mlmc_gradient_estimate=load('author_gradient','mlmc_optim/gradient.py').mlmc_gradient_estimate
MLMCBatcher=load('author_batcher','mlmc_optim/batcher.py').MLMCBatcher

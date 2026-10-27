"""Explicitly load the frozen local analyzer, avoiding legacy sys.path shadowing."""
import pathlib,importlib.util
R=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('publication_formal_mg_analysis',R/'analyze.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.main()

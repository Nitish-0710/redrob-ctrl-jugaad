"""
run_tests_unittest.py
=====================
A dynamic test runner that discovers and runs pytest-style functions 
using Python's built-in unittest framework. No external dependencies required.
"""

import unittest
import types
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import test modules
import tests.test_parser as test_parser
import tests.test_validators as test_validators
import tests.test_candidate_features as test_candidate_features
import tests.test_scorer as test_scorer
import tests.test_reasoning as test_reasoning

modules = [
    ("test_parser", test_parser),
    ("test_validators", test_validators),
    ("test_candidate_features", test_candidate_features),
    ("test_scorer", test_scorer),
    ("test_reasoning", test_reasoning)
]

def make_test_case_class(name, module):
    """Dynamically creates a unittest.TestCase subclass for a module's test functions."""
    test_methods = {}
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if attr_name.startswith("test_") and isinstance(attr, types.FunctionType):
            # Create a TestCase method wrapping the function call
            def make_test_method(func=attr):
                return lambda self: func()
            test_methods[attr_name] = make_test_method()
            
    return type(name, (unittest.TestCase,), test_methods)

def main():
    suite = unittest.TestSuite()
    for name, module in modules:
        test_case_cls = make_test_case_class(name, module)
        tests = unittest.defaultTestLoader.loadTestsFromTestCase(test_case_cls)
        suite.addTests(tests)
        
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    main()

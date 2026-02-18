#%%
import numpy as np
import pandas as pd
import warnings
import contextlib
import os
import functools
import tempfile
from pathlib import Path

def is_iterable(obj, allow_strings=False):
    """
    Check if an object is iterable. Note that empy arrays or lists are considered iterable, so you may want to check also for the length of the object. Also, we do not consider strings as iterable by default, although their elements can be accessed by index.
    """
    if isinstance(obj, dict):
        return False
    elif isinstance(obj, str):
        return True if allow_strings else False
    else:
        try:
            _ = obj[0]
            return True
        except IndexError: # must be an empty list or array
            return False
        except TypeError:
            return False

def input_prompt(prompt_msg, default=None, assert_type=None, allow_retry=False):

    """
    Ask the user for input, and return the default value if the input is empty.
    Args:
        - prompt_msg (str): the prompt to show to the user
        - default: the default value to return if the user input is empty
        - assert_type (list, tuple): the type(s) to assert the input to be
    Returns:
        - value: the user input (converted as a literal value if possible, and as a string if not), or the default value if the user input is empty
    Example:
        x = input_prompt('Enter a scalar value (defaults to 0): ', 0, assert_type=(int, float)); print(x)
        y = input_prompt('Enter an array: ', [], assert_type=(list, tuple, range, np.ndarray)); print(y)
        z = input_prompt('Enter anything: ', assert_type=None); print(z)
    """

    user_input = input(prompt_msg)
    
    def _check(assert_type):
        
        if user_input == '':
            value = default
        else:
            try:

                # import ast
                # Try to parse the input as a literal value
                # value = ast.literal_eval(user_input)
                value = eval(user_input)

            except (ValueError, SyntaxError, NameError):
                value = user_input

            if assert_type is not None:
                if not is_iterable(assert_type):
                    assert_type = (assert_type,)
                if type(assert_type) is not tuple:
                    assert_type = tuple(assert_type)
                assert isinstance(value, assert_type), f'Input must be of type {[x.__name__ for x in assert_type]}'
                
        return value
    
    if allow_retry:
        
        try:
            value = _check(assert_type)
        except AssertionError as e:
            print(e)
            return input_prompt(prompt_msg, default, assert_type, allow_retry)
            
    else:
        value = _check(assert_type)
                
    
    return value
# CallableDict

A dictionary that can be called like a function.

This class extends the built-in dict class and adds a __call__ method,
which allows it to be used as a callable object. This is particularly useful
for making the result of build_preprocessor callable, so users can do
preprocessor(test_input) instead of preprocessor["model"](test_input).

When called, it will try to invoke the "model" key if it exists, passing all
arguments and keyword arguments to that function.

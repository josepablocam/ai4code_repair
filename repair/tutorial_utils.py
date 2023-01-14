from IPython.display import Code
import inspect

def show_code(obj):
    return Code(inspect.getsource(obj), language='python')
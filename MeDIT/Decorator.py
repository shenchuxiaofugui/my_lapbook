"""
MeDIT.Decorator
Provide some decorator to help show and store figures and dict informations.

author: Yang Song [songyangmri@gmail.com]
All right reserved
"""

from functools import partial

import pandas as pd
import matplotlib.pyplot as plt


def attach_wrapper(obj, func=None):
    if func is None:
        return partial(attach_wrapper, obj)
    setattr(obj, func.__name__, func)
    return func


def figure_decorator(store_path='', show=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            c = func(*args, **kwargs)
            if store_path:
                plt.tight_layout(True)
                if store_path[-3:] == 'jpg':
                    plt.savefig(store_path, dpi=300, format='jpeg')
                elif store_path[-3:] == 'eps':
                    plt.savefig(store_path, dpi=1200, format='eps')
            if show:
                plt.show()
            plt.close()
            return c

        @attach_wrapper(wrapper)
        def set_show(new_show):
            nonlocal show
            show = new_show

        @attach_wrapper(wrapper)
        def set_store_path(new_store_path):
            nonlocal store_path
            store_path = new_store_path
        return wrapper
    return decorator


def dict_decorator(store_path='', show=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            c = func(*args, **kwargs)
            if store_path and store_path.endswith('csv'):
                df = pd.DataFrame(c, index=[0])
                df.to_csv(store_path)
            if show:
                print(c)
            return c

        @attach_wrapper(wrapper)
        def set_show(new_show):
            nonlocal show
            show = new_show

        @attach_wrapper(wrapper)
        def set_store_path(new_store_path):
            nonlocal store_path
            store_path = new_store_path
        return wrapper
    return decorator

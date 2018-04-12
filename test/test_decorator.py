def print_args(func):
  def wrapped_func(*args, **kwargs):
    print(func.__name__, func.__doc__)
    for a in args:
      print(a)
    for k, v in kwargs:
      print(k, v)
    return func(*args, **kwargs)
  return wrapped_func


@print_args
def add(x: int, y: int):
  """:Add two integers"""
  return x + y


print(add(1, 2))

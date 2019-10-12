from datetime import datetime


def measure_performance(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = datetime.now()
            func(*args, **kwargs)
            end = datetime.now()
            print(f"{name}: {(end - start).__str__()}")
        return wrapper
    return decorator


@measure_performance('ADDING UP')
def do_some_stuff(x, y, z):
    print(x + y - z)

do_some_stuff(434,235231452345324,43523452435)
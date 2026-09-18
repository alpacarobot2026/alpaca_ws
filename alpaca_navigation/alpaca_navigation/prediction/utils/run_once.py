"""
Decorator: Makes the function only run once
"""

def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper



if __name__ == "__main__":
    
    class test_:
        def __init__(self) -> None:
            self.num = 0

        @run_once
        def my_function(self, bar):
            self.num += bar
    
        def __str__(self) -> str:
            return str(self.num)

    test_0 = test_()
    for _ in range(5):
        test_0.my_function(1)
        # print(test_0)

    print("final",test_0)
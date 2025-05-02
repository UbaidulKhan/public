def greet(name=None):
    if name is None:
        name = "Joe"  # Assign a default value if no argument is provided
    print("Hello, " + name + "!")

var_name = "Alice"
greet(name=var_name)

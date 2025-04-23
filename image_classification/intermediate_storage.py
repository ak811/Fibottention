# intermediate_storage.py

# Define a variable to hold the intermediate value
saved_x = None

def save_intermediate_x(x):
    global saved_x
    saved_x = x.clone().detach()  # Store a detached copy of x

def get_intermediate_x():
    return saved_x  # Retrieve the saved value
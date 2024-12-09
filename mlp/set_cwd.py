def set_cwd():
    import os
    with open('mlp/cwd.txt', 'w') as file:
        file.write(os.getcwd())
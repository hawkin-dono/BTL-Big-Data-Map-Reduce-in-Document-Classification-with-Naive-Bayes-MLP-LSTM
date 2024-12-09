def set_cwd():
    import os
    with open('bayes_hadoop/cwd.txt', 'w') as file:
        file.write(os.getcwd())
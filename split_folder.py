import os, os.path, shutil

def split_folder(folder_path, num_folders):
    '''
    Splits the contents of a folder into smaller subfolders.
    Args:   folder_path (str): path to folder to be split
            num_folders (int): number of subfolders to be created
    '''
    docs = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for i in range(num_folders):
        new_path = os.path.join(folder_path, f'subset_{i}')
        if not os.path.exists(new_path):
            os.makedirs(new_path)

    for doc in docs:
        index = docs.index(doc)
        subset = int(index * num_folders / len(docs))
        old_doc_path = os.path.join(folder_path, doc)
        new_doc_path = os.path.join(os.path.join(folder_path, f"subset_{subset}"), doc)
        shutil.move(old_doc_path, new_doc_path)
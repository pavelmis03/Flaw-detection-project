'''
Функции вывода информации
'''


def printAnnotationPath(all_files, path_an):
    print("Пути файлов аннотаций: ")
    last_dir = ""
    for anno in all_files:
        t = anno[len(path_an) + 1:anno.index("/", len(path_an) + 1)]
        if (last_dir != t):
            print("\n\n", path_an + t)
            last_dir = t
        print(anno[anno.index(t) + len(t) + 1:])

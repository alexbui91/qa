import properties as p
import pickle 
import os.path as path
import os
import codecs


def load_glove(use_index=False):
    if use_index:
        word2vec = []
    else:
        word2vec = {}
    print("==> loading glove")
    if not p.glove_file:
        with open(("%s/glove.6B.%id.txt") % (p.glove_path, p.embed_size)) as f:
            for line in f:
                l = line.split()
                if use_index:
                    word2vec.append(map(float, l[1:]))
                else:
                    word2vec[l[0]] = map(float, l[1:])
    else:
        with open(("%s/%s") % (p.glove_path, p.glove_file)) as f:
            for line in f:
                l = line.split()
                if use_index:
                    word2vec.append(map(float, l[1:]))
                else:
                    word2vec[l[0]] = map(float, l[1:])

    print("==> glove is loaded")

    return word2vec


def validate_path(name):
    paths = name.split("/")
    tmp = ""
    for e, folder in enumerate(paths):
        tmp += folder
        if e != 0:
            tmp += "/"
        if not path.exists(tmp):
            os.path.makedirs(tmp)


def save_file(name, obj, use_pickle=True):
    validate_path(name)
    with open(name, 'wb') as f:
        if use_pickle:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        else: 
            f.write(obj)


def save_file_utf8(name, obj):
    with codecs.open(name, "w", "utf-8") as file:
        file.write(u'%s' % obj)


def load_file(pathfile, use_pickle=True):
    if path.exists(pathfile):
        if use_pickle:
            with open(pathfile, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(pathfile, 'rb') as file:
                data = file.readlines()
        return data 


def load_file_utf8(pathfile):
    if path.exists(pathfile):
        with codecs.open(pathfile, "r", "utf-8") as file:
            data = file.readlines()
        return data 


def check_file(pathfile):
    return path.exists(pathfile)


def intersect(c1, c2):
    return list(set(c1).intersection(c2))


def sub(c1, c2):
    return list(set(c1)- set(c2))


def update_progress(progress, sleep=0.01, barLength=20):
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
    time.sleep(sleep)
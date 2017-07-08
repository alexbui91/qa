import os

folders = ['en', 'en-10k']
for f in folders:
    files = os.listdir("data/%s" % f)
    for fl in files:
        name = fl.split("_")
        new_name = "%s_%s" % (name[0], name[-1])
        os.rename("data/%s/%s" % (f, fl), "data/%s/%s" % (f, new_name))

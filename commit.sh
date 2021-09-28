#!/bin/sh
newv=`python -c 'v={}; exec(open("servalcat/__init__.py").read(), v); v=v["__version__"].split("."); print(".".join(v[:-1])+".{}".format(int(v[-1])+1))'`
sed -i "s/__version__.*/__version__ = '$newv'/" servalcat/__init__.py
git add servalcat/__init__.py
git commit

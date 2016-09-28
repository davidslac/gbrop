from __future__ import print_function

import os
import sys

pkgs2lnk = {
    'psmlearn':'/reg/neh/home/davidsch/github/davidslac/psana-mlearn/master',
    'h5minibath':'/reg/neh/home/davidsch/github/davidslac/h5-mlearn-minibatch/v0.0.1'
}

data2lnk = {
    'vgg16_weights.npz':'/reg/neh/home/davidsch/mlearn/vgg_xtcav/vgg16_weights.npz',
}

PKGDIR = os.path.join(os.path.abspath(os.path.split(__file__)[0]),'pypkgs')
DATADIR = os.path.join(os.path.abspath(os.path.split(__file__)[0]),'data')

for direc in [PKGDIR, DATADIR]:
    if not os.path.exists(direc):
        os.mkdir(direc)

for lnk2loc, basedir, add2path in zip([pkgs2lnk, data2lnk],
                                      [PKGDIR, DATADIR],
                                      [True,False]):
    for lnk, direc in lnk2loc.iteritems():
        symlnk = os.path.join(basedir, lnk)
        if not os.path.exists(symlnk):
            assert os.path.exists(direc), "for lnk=%s direc doesn't exist: %s" % (lnk, direc)
            os.symlink(direc, symlnk)
        if add2path:
            if symlnk not in sys.path:
                sys.path.append(symlnk)

if __name__ == '__main__':
    if len(sys.argv)>=2 and sys.argv[1].lower().strip()=='clean':
        import shutil
        for direc in [PKGDIR, DATADIR]:
            if os.path.exists(direc):
                if 'y' == raw_input("remove directory %s? [y/n]" % direc).strip().lower():
                    shutil.rmtree(direc)
    else:
        print("give argument clean to remove the pypkg and data dirs")


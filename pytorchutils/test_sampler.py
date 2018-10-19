import utils

from sys import argv

#Quick script for testing that your samples look
# the way you expect them to

SAMPLER_FNAME = argv[1]
AUGMENTOR_FNAME = argv[2]
NUM_SAMPLES = int(argv[3]) if len(argv) > 3 else 3


Sampler = utils.load_source(SAMPLER_FNAME).Sampler
Augmentor = utils.load_source(AUGMENTOR_FNAME).get_augmentation


patchsz = (1,18,160,160)

s = Sampler("~/seungmount/research/agataf/datasets/", patchsz, 
            vols=["vol501"], aug=Augmentor(True))

for i in range(NUM_SAMPLES):

    samp = s()

    for (k,v) in samp.items():
        for j in range(v.shape[0]):
          utils.write_h5(v[j,:,:,:], "tests/sample{}_{}{}.h5".format(i,k,j))


"""
A quick script for testing that your samples look the way you expect.

Nicholas Turner <nturner@cs.princeton.edu>, 2017-9
"""
import os
import argparse

from pytu import utils


SAMPLE_DIRNAME = "samples"


def main(sampler_fname, num_samples, *args, **kwargs):
    sample_iter = load_iter(sampler_fname, *args, **kwargs)

    for i in range(num_samples):
        sample = next(sample_iter)
        write_sample(sample, i)


def load_iter(sampler_fname, *args, **kwargs):
    sampler_class = utils.load_source(sampler_fname, "S").Sampler
    return iter(sampler_class(*args, **kwargs))
    

def write_sample(sample, sample_num):
    """Writes a sample (dict) as separate files for viewing"""
    if not os.path.isdir(SAMPLE_DIRNAME):
        os.makedirs(SAMPLE_DIRNAME)

    for (k, v) in sample.items():
        utils.write_h5(v, f"{SAMPLE_DIRNAME}/sample{sample_num}_{k}.h5")


def parse_other_args(other_stuff):
    args, kwargs = [], {}

    for arg in other_stuff:
        if '=' not in arg:  # std arg
            try:
                args.append(eval(arg))
            except SyntaxError as e:
                args.append(arg)  # raw string arg
        else:
            fields = arg.split('=')
            assert len(fields) == 2, f"improper arg: {arg}"
            kwargs[eval(fields[0])] = eval(fields[1])

    return args, kwargs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument("sampler_fname")
    parser.add_argument("num_samples", type=int)
    parser.add_argument("other_args", nargs='*')

    args = parser.parse_args()

    sampler_args, sampler_kwargs = parse_other_args(args.other_args)
    main(args.sampler_fname, args.num_samples,
         *sampler_args, **sampler_kwargs)

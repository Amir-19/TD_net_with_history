import sys


# code from: https://gist.github.com/marqueewinq/b78c1a92b357b4aa95281e7721eba625
# print iterations progress


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    format_str = "{0:." + str(decimals) + "f}"
    percent = format_str.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

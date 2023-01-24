import bisect


def age_bucket(x):
    return bisect.bisect_left([18, 25, 35, 45, 55, 65], x)

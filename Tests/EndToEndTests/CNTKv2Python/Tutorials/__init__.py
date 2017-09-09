import numpy
def _all_close_or_less(result, expect, atol):
    df = (result - expect)
    f1 = df < 0
    f2 = numpy.abs(df) <= tol
    return numpy.all(f1 | f2)
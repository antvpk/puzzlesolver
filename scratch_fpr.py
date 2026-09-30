import math

def poisson(l, x):
    return math.exp(-l) * (l**x) / math.factorial(x)

def fpr(l, k, b):
    total = 0.0
    for x in range(0, 100):
        p_x = 1.0 - math.exp(-k * x / b)
        f_x = p_x ** k
        prob = poisson(l, x)
        total += prob * f_x
    return total

print(fpr(30891132 / 4194304, 20, 512))

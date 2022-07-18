"""Comprehension

listComprehension = [expression for i in iterable if expression else expression] etc.
"""

from functools import reduce
result = []

for i in range(1, 11):
    result.append(i ** 2)

print(result, end=' ')
# Simdi bu yapiyi Comprehension ile yapalim
print('\n************************************************')

result2 = [i ** 2 for i in range(1, 11)]
print(result2)

print('*************************************************')

result3 = {i: i ** 3 for i in range(1, 11)}
print(result3)

print('*************************************************')

faktoriyel = reduce(lambda x, y: x * y, [i for i in range(1, 6)])
print(faktoriyel)

print('*************************************************')

x = ["MerhaBA", "PYTHON", "pROGRAMLAMA", "Dili"]
y = [i.title() for i in x]
print(' '.join(y))

print('*************************************************')

arr = ["kayak", "adana", "yapay", "kek","urfa","hatay"]
arr2 = list(filter(lambda i: i == i[::-1], arr))
print(arr2)

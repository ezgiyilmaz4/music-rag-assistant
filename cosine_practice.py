import math

def dot_product(a,b):
    total = 0
    for i in range(len(a)):
        total += a[i]*b[i]
    return total


def magnitude(a):
    total = 0
    for i in range(len(a)):
        total += a[i]**2
    total = math.sqrt(total)
    return total  

def cosine_similarity(A,B):
    cs=ddot_product(A,B) / (magnitude(A) * magnitude(B))
    return cs 



vec1= [1,2,3]
vec2 = [4,5,6]
print(dot_product(vec1,vec2))
print(magnitude(vec1))
print(cosine_similarity(vec1, vec2))

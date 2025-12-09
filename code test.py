def min_pieces(original, desired):
    if original == desired:
        return 0
    res = []
    for i,j in groupby(a,lambda x: x == b):
        if not i:
            res.append(list(j))
    print(res)
    
    

original = [1, 4, 3, 2]
desired = [1, 2, 4, 3]
print(min_pieces(original, desired))
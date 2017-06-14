

def equal_dictionaries(dic1, dic2):

    # compare keys
    keys1 = set([k for k in dic1.keys()])
    keys2 = set([k for k in dic2.keys()])

    if keys1 != keys2:
        print("Different keys")
        return False

    for k in dic1.keys():
            if isinstance(dic1[k], dict) and isinstance(dic2[k], dict):
                if not equal_dictionaries(dic1[k], dic2[k]):
                    return False
            else:
                if dic1[k] != dic2[k]:
                    print("Different values at key {}".format(k))
                    return False
    return True

if __name__ == '__main__':

    d1 = {'a': {'b': {'cs': 10}, 'd': {'cs': 20}}}
    d2 = {'a': {'b': {'cs': 30}, 'd': {'cs': 20}}, 'newa': {'q': {'cs': 50}}}
    d3 = {'a': {'b': {'cs': 10}, 'd': {'cs': {'w':1}}}}
    d4 = {'a': {'b': {'cs': 10}, 'd': {'cs': [1,2,3]}}}

    print(equal_dictionaries(d1, d1))
    print(equal_dictionaries(d1, d2))
    print(equal_dictionaries(d1, d3))
    print(equal_dictionaries(d1, d4))


def ToSqlField(s):
    filter1 = s.replace(".","_").replace(" ","_")
    filter2 = ''.join(e for e in filter1 if e.isalnum())
    return filter2
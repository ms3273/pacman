strings = ['Some string','Art','Music','Artificial Intelligence']

proc_string = [x.lower() for x in strings if len(x) > 5]

print(proc_string)

#print [x.lower() for x in strings if len(x) > 5]
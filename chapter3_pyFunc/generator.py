'''use the yield, the reason why use the generator, even when compared with the list comprehension, main reason is for the 
memory efficiency'''
list_=[x**2 for x in list(range(10)) if x%2!=0] 
print(list_)

print('-'*50) 

gen=(x**2 for x in list(range(10)) if x%2!=0) 
for i in gen:
    print(i)   
#----------------------------------------------------

def gen(list_):
    for i in list_:
        yield i 

list_gen= gen(list(range(10)))
next(list_gen) #print until StopIteration error 

def gen(list_):
    for i in list_:
        yield i 

list_gen= gen(list(range(10))) 
for j in list_gen:
    print(j) 

#__iter__ magic method usage 
class Name:
    def __init__(self, name, city):
        self.name= name
        self.city= city 
        
    def __iter__(self):
        for name, city in zip(self.name, self.city):
            yield name, city 
            
people= Name(('chloe','nathan', 'emma'), ('shanghai', 'lille', 'newyork')) 
for n, c in people:
    print('{} lives in {}'.format(n,c))  


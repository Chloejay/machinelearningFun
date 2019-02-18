#for the decorator it takes me a while to understand, sometimes I felt I understand but the another I have no clue to make it right. So I only keep one sentence in my mind, decorator is the function return object, which is function, sounds is the higher order function

#key concept support recursion, think this concept as the packaged box, like the wrapper or think decorator as the shortcut to call the functions 
def outpack(word):
    def wrapper():
        print(word) #free variable to use outside the function scope 
    return wrapper() 
de=  outpack('Das ist Schone')
eng= outpack('It\'s Ok')

#to change variable to the function, to use @decorator to `alter or extend` new functionality 
def outpack (func):
    def wrapper(*args, **kwargs): 
        return func(*args, **kwargs) 
    return wrapper 

@outpack
def add():
    print('schone') 
    
@outpack
def call():
    print('cool') 

@outpack
def multiply(x,y):
    print(x*y)

#@property is another property to lighten class code 
class Travelling():
    def __init__(self, city, people):
        self.city= city
        self.people= people
    @property 
    def where(self):
        print('Travel to {}'.format(self.city)) 

    def who(self):
        print('{} is gone to {}'.format(self.people, self.city))

t= Travelling('mars', 'chloe')

if __name__=='__main__':
    de
    eng 
    add()
    call()  
    multiply(9,10) 
    t.where
    t.who() 



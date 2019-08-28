'''let's create  a story for the class, for class is like the relational database, hierarchy tree (like parents and daughter (superclass and subclass), or company and employee), who share the similar attributes, also called properties, but the differential part identify the individualism. BUT the `DRY concept` makes code fun and also make us enjoy laziness. 
So here we use inheritance, and magic methods __init__ and __call__, __repr__ , and all the related, use help(__dir__(func)) to check the built-in *magic methods. ''' 

#concept explaintion 
'''why need to use super() and MRO concept, resource: The wonders of cooperative inheritance by https://www.artima.com/weblogs/viewpost.jsp?thread=281127 
things to know about super essay, as per the OOP function, class is the first class object in the python and it can be used as any objects. When started to use the class, 
it's useful to use the init constructor to define some parameters which can be used inside the wrapped function, very typical to use self, which is equal to that in the JavaScript,
 but another useful way to use class, is for thee inheritance, which can avoid to rewrite the same function twice. when inherit the functions from the superclass(parent class) to the subclass (child class), 
 the hierarchy tree is needed to use the concept MRO (Method Resolution Order) and super().init() is used in the Python3. It subclass dispatch the super call method to the 
 superclass, the logic follows the MRO algorithm. use super only when the methods in a cooperative chain have consistent signature even super is not the keyword yet, but avoid to use super as the fn name'''

#emma is one frequently used name by me and Berlin is my fave city by far 

class Minions():
    def __init__(self,  firstName, lastName): #constructor for properties (or __init__(self) to df the varibales) 
        self.color= 'yellow'
        self.firstName= firstName 
        self.lastName=lastName
        self.age=0.1 
        self.list=[] 

    def callMinions(self): #call self 
        print('we call the {}.{}, whose color is {}'.format(self.lastName, self.firstName, self.color))
    
class Minions_1(Minions):
    def __init__(self, firstName, lastName, city):
        super().__init__(firstName,lastName)
        self.city= city

    def travel(self):
        print('{}.{} travels to {},  their age are all {}'.format(self.firstName,self.lastName, self.city, self.age+10))  
    
    def __repr__(self):
        print (self.firstName)  
    def __call__(self):
        print(self.__dir__())  

if __name__== '__main__':
    eifi= Minions('eifi','tower') #instantiate the real-data 
    eifi.callMinions()
    emma= Minions_1('emma','emily', 'berlin') 
    emma.travel() 
    emma.callMinions() #behave Minions does, sometimes think method as attribute 
    #magic methods usage just like built-in methods 
    emma.__repr__() 
    emma.__call__() 

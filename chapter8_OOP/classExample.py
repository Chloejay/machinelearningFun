#let's imagine a story for the class, for class is like the relational database, hierarchy tree (like parents and daughter (superclass and subclass), or company and employee), who share the similar attributes, also called properties, but the differential part identify the individualism. BUT the `DRY concept` makes code fun and also make us enjoy lazy. So here we use inheritance, and magic methods __init__ and __call__, __repr__ , and all the related, use help(__dir__(func)) to check the built-in *magic methods. 

#Dispicable me minions as the base 
class Minions():
    def __init__(self,  firstName, lastName): 
        #constructor for properties 
        self.color= 'yellow'
        self.firstName= firstName
        self.lastName=lastName
        self.age=0.1
        self.list=[] 

    def callMinions(self):
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






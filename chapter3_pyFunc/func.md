<p>why I include this chapter on the Python ML/DS self reading book, for Python is about light code and I learnt Javascript before Python, when comparing the functionalities between them. Remembered when I first knew inheritance concept is 2017 when I was learning React.js. In OOP concept, eveything is object, which should be treated as the first-class citizen in the progrmming zone. Such as the concept of treat the function as value. On this chapter I will summarize what I learnt, including main concept of composition, inheritance, polymorphism, class, generator, iterator (list/dict/set comprehension),built-in functions, recursion, decorator, regexp usage, exceptions and some idioms.</p> <br/> 

two idioms I liked: 
- Use return to evaluate expressions as well as return values
- Always use self or a @classmethod when referring to a class’s attributes  

This session aims to point out some tricks or pitfalls that make the code more miantable and easy to debug.  
- Python’s assert statement is a debugging aid that tests a condition as an internal self-check in your program, especially for unittest case; 
- Asserts should only be used to help developers identify bugs. They’re not a mechanism for handling run-time errors; 
- Asserts can be globally disabled with an interpreter setting. 

with statement: allows you to abstract away most of the resource handling logic for the data I/O which can aviod the try .. finally stmt to make the code more verbose. The with statement can make code that deals with system resources
more readable. It also helps you avoid bugs or leaks by making it practically impossible to forget to clean up or release a resource when it’s no longer needed. 

Optional use the context manager 
from contextlib import contextmanager
`
@contextmanager
def managed_file(name):
try:
    f = open(name, 'w')
    yield f
finally:
    f.close() 
    
with managed_file('hello.txt') as f:
    f.write('hello, world!')
    f.write('bye now') 
` 

underscore for the varibale and method/func
_var: that a variable or method starting with a single underscore is intended for internal use. the difference for the private and public variables, but in the real case, variables with/out the _ doesn't have any diff(the convention in the python community), only the difference shows on the on the func, of use def _func, when call this func out, the output will raise the NameError: "name '_func' is not defined". 

when dealing with the reserved case words in the python or any other programming language, use var_ to make this varibale can be useful. 
`
def count(list_):
    for i in list_:
            print(i) 
` 
            
__var will face the name mangling, the old name will not be showed in the attributes when use the dir(method/instance); if only use the '_' then it can be translated as the "not_care" var, or named as the temp var 


#exception is everywhere in Python, for if something you are not sure, you can use the exception to make it through 

def my_generator():
    try:
        yield 'something'
    except ValueError as e:
        print(e) 
        yield 'dealing with the exception'
    finally:
        print ("ok let's clean") 




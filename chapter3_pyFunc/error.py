#exception is everywhere in Python, for if something you are not sure, you can use the exception to make it through 
#error handling code and error checking,  Programs sometimes fail, especially during development, and if you can’t determine why failures occur, you’re flying blind.

def my_generator():
    try:
        yield 'something'
    except ValueError as e:
        print(e) 
        yield 'dealing with the exception'
    finally:
        print ("ok let's clean") 


def mergeList(listA, listB):
    new_list=list() 
    a= 0
    b=0 

    while a <len(listA) and b< len(listB):
        if listA[a]<listB[b]:
            new_list.appen(listA)
            a+=1 
        else:
            new_list.appen(listB)
            b+=1 
            
    return new_list 


#iter() and __iter__ magic method 

#used with generator and comprehension 
'''[ fn(var) for var in iterator_list if conditions], the same logic for the {:} dict comprehension and also for the set ''' 

# df.iterrows()  

def fn():
    x=list()
    y=list()
    for idx, row in df.iterrows():
        x.append(df(row))
        y.append(df[idx])
    return (x, y) #return as the tuple  


    
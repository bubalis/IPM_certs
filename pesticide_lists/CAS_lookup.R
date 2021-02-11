library (webchem)


CAS_CIDretriever <-function(name){
  response=get_cid(name)
  if (is.na(response)){
    return (NA)
  }
  else{
    r=cts_convert(response[[2]], from='pubchem cid', to="cas")
    return (r[[1]])}
  
}

multi_query<- function(name){
  print(name)
  
  response<- aw_query(name, from = 'name')
  
  if (is.na(response[[1]])){
    print('Failed Trying again')
    response<-ci_query(name, from ='name' )
  }
  
  else{return(response[[1]]$cas) }
  
  if (is.na(response[[1]])){
    print('Failed Trying again')
    response<-CAS_CIDretriever(name)}
  else{return(response[[1]]$cas)}
  
  if (is.na(response[[1]])){return (NA)}
  
  
  
  else{
    return (response) 
    
  }
}
myargs<-commandArgs(trailingOnly=TRUE)

name=myargs[1]

cat(multi_query(name))




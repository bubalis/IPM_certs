library (webchem)
library (jsonlite)



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
  







for (file in list.files() ){
  if (grepl('.txt', file, fixed = TRUE)) {
    data<- read.delim(file, sep='\n', header=FALSE)
    out<-  list()
    
    for (name in data$V1){
      if (!name %in% names(out))
        res=NULL
      try(res<-multi_query(name))
      out[name]<-res
      
    }
    
    sink(paste('keys', file, sep='_'))
    print(toJSON(out))
    sink()
  }
}

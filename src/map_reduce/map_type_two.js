function(){

  for (var word = 0; word < this.content.length; word++)
  {
       var key = this.content[word];
       var value = { classX:this.classX, classY:this.classY };

       emit(key,value);
  }

};

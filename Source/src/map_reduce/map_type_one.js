function() {

    var key = 'doc';
    var value = {clX:0, clY:0, V:this.content.length};

    if (this.classX==1){
    	value.clX=this.content.length;
    }
    else {
    	value.clY=this.content.length;
    }

    emit (key,value)
}

function() {

    var key = 'doc';
    var value = { a:0, b:0, c:0, d:0 };

    if (this.classX == 1){

        if (this.predClassX == 1){
    	   value.a = 1;
        }
        else {
            value.b = 1;
        }

    }
    else {

        if (this.predClassX == 1) {
        	value.c = 1;
        }
        else {
            value.d = 1;
        }

    }

    emit (key,value)
}
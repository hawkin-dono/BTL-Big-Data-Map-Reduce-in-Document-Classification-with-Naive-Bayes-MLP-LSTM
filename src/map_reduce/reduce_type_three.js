function(key, values) {
    var total = {a:0, b:0, c:0, d:0};
    values.forEach(function(value){
    	total.a +=value.a;
    	total.b +=value.b;
    	total.c +=value.c;
    	total.d +=value.d;
    });
    return total;
    }
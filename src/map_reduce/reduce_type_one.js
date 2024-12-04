function(key, values) {
    var total = {clX:0, clY:0, V:0};
    values.forEach(function(value){
    	total.clX +=value.clX;
    	total.clY +=value.clY;
    	total.V +=value.V;
    });
    return total;
    }
function(key, values)
{
    wordCount={classX:0,classY:0};
    values.forEach(function(value){ wordCount.classX += value.classX;
                                    wordCount.classY += value.classY;
    });
    return wordCount;


};
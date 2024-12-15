def get_total_counts_pipeline():
    return [
            { #mapper
            "$project": {
                "cl0": {
                    "$cond": {
                        "if": { "$eq": ["$class0", 1] },
                        "then": { "$size": "$content" },
                        "else": 0
                    }
                },
                "cl1": {
                    "$cond": {
                        "if": { "$eq": ["$class1", 1] },
                        "then": { "$size": "$content" },
                        "else": 0
                    }
                },
                "cl2": {
                    "$cond": {
                        "if": { "$eq": ["$class2", 1] },
                        "then": { "$size": "$content" },
                        "else": 0
                    }
                },
                "cl3": {
                    "$cond": {
                        "if": { "$eq": ["$class3", 1] },
                        "then": { "$size": "$content" },
                        "else": 0
                    }
                },
                "V": { "$size": "$content" }
            }
        },
        { #reducer
            "$group": {
                "_id": "doc",
                "cl0": { "$sum": "$cl0" },
                "cl1": { "$sum": "$cl1" },
                "cl2": { "$sum": "$cl2" },
                "cl3": { "$sum": "$cl3" },
                "V": { "$sum": "$V" }
            }
        },
        { #output
            "$out": "TotalCounts"
        }
    ]
    
    
def get_word_counts_pipeline():
    return [
        # Step 1: Unwind the content array to create a document for each word
    { 
        "$unwind": "$content" 
    },
    
    # Step 2: Project the required fields with the exact structure
    {
        "$project": {
            "_id": 0,
            "word": "$content",
            "value": {
                "class0": "$class0",
                "class1": "$class1",
                "class2": "$class2",
                "class3": "$class3",
            }
        }
    },
    
    # Step 3: Group by word (equivalent to the reduce phase)
    {
        "$group": {
            "_id": "$word",
            "class0": { "$sum": "$value.class0" },
            "class1": { "$sum": "$value.class1" },
            "class2": { "$sum": "$value.class2" },
            "class3": { "$sum": "$value.class3" },
        }
    },
    
    # Step 4: Reshape the output to match the original format
    {
        "$project": {
            "_id": 1,
            "value": {
                "class0": "$class0",
                "class1": "$class1",
                "class2": "$class2",
                "class3": "$class3",
            }
        }
    },
    
    # Step 5: Write to output collection
    {
        "$out": "WordCounts"
    }
    ]
    
def get_statistics_result_pipeline():
    return [
        {
        "$project": {
            "emit": {
                "key": "doc",
                "value": {
                    "class0_true": {
                        "$cond": [
                            { "$eq": ["$class0", "$predClass0"]},
                            1,
                            0
                        ]
                    },
                    "class0_false": {
                        "$cond": [
                            { "$ne": ["$class0", "$predClass0"]},
                            1,
                            0
                        ]
                    },
                    "class1_true": {
                        "$cond": [
                            { "$eq": ["$class1", "$predClass1"]},
                            1,
                            0
                        ]
                    },
                    "class1_false": {
                        "$cond": [
                            { "$ne": ["$class1", "$predClass1"]},
                            1,
                            0
                        ]
                    },
                    "class2_true": {
                        "$cond": [
                            { "$eq": ["$class2", "$predClass2"]},
                            1,
                            0
                        ]
                    },
                    "class2_false": {
                        "$cond": [
                            { "$ne": ["$class2", "$predClass2"]},
                            1,
                            0
                        ]
                    },
                    "class3_true": {
                        "$cond": [
                            { "$eq": ["$class3", "$predClass3"]},
                            1,
                            0
                        ]
                    },
                    "class3_false": {
                        "$cond": [
                            { "$ne": ["$class3", "$predClass3"]},
                            1,
                            0
                        ]
                    },
                }
            }
        }
    },
    
    {
    "$group": {
        "_id": "$emit.key",
        "value": {
            "$accumulator": {
                "init": """function() { return {class0_true: 0, class0_false: 0, class1_true: 0, class1_false: 0, 
                                    class2_true: 0, class2_false: 0, class3_true: 0, class3_false: 0}; }""",
                "accumulate": """function(state, value) { return {class0_true: state.class0_true + value.class0_true, 
                                    class0_false: state.class0_false + value.class0_false, class1_true: state.class1_true + value.class1_true, 
                                    class1_false: state.class1_false + value.class1_false, class2_true: state.class2_true + value.class2_true, 
                                    class2_false: state.class2_false + value.class2_false, class3_true: state.class3_true + value.class3_true, 
                                    class3_false: state.class3_false + value.class3_false}; }""",
                "accumulateArgs": ["$emit.value"],
                "merge": """function(state1, state2) { return {class0_true: state1.class0_true + state2.class0_true, 
                class0_false: state1.class0_false + state2.class0_false, class1_true: state1.class1_true + state2.class1_true, 
                class1_false: state1.class1_false + state2.class1_false, class2_true: state1.class2_true + state2.class2_true, 
                class2_false: state1.class2_false + state2.class2_false, class3_true: state1.class3_true + state2.class3_true, 
                class3_false: state1.class3_false + state2.class3_false}; }""",
                "lang": "js"
                }
            }
        }
        },
        {
            "$out": "Results"
        }
    ]

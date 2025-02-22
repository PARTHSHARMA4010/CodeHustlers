dataset = [
    {
        "skin_analysis": {
            "tone": "light",
            "texture": "smooth",
            "dark_circles": "present",
            "spots": [{"location": "x=10, y=200", "size": 4}]
        },
        "nail_analysis": {
            "color": {"RGB": [90, 100, 110], "Hex": "#5a646e"},
            "texture": {"edge_intensity": 0.3},
            "shape": {"shapes_detected": ["Round"]}
        },
        "outputs": {
            "disease": "vitiligo",
            "dosha_analysis": {
                "vata": "mildly imbalanced",
                "pitta": "balanced",
                "kapha": "slightly imbalanced"
            },
            "observations": [
                "Skin pigmentation changes indicate mild Kapha imbalance.",
                "Round nail shape suggests a balanced Vata condition."
            ],
            "recommendations": {
                "diet": ["Increase intake of leafy greens.", "Avoid fried and spicy foods."],
                "lifestyle": ["Practice yoga to enhance circulation.", "Use sunscreen daily."],
                "herbal_remedies": ["Neem capsules for skin health.", "Ashwagandha for stress relief."]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "medium",
            "texture": "normal",
            "dark_circles": "absent",
            "spots": [{"location": "x=15, y=180", "size": 3}]
        },
        "nail_analysis": {
            "color": {"RGB": [80, 85, 90], "Hex": "#50555a"},
            "texture": {"edge_intensity": 0.1},
            "shape": {"shapes_detected": ["Oval"]}
        },
        "outputs": {
            "disease": "anemia",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "low",
                "kapha": "imbalanced"
            },
            "observations": [
                "Pale nails and skin indicate low Pitta activity.",
                "Nail shape suggests Kapha imbalance."
            ],
            "recommendations": {
                "diet": ["Consume iron-rich foods like spinach.", "Avoid cold and heavy meals."],
                "lifestyle": ["Regular exercise to improve circulation.", "Maintain consistent sleep patterns."],
                "herbal_remedies": ["Triphala to improve digestion.", "Ashwagandha to boost energy."]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "dark",
            "texture": "rough",
            "dark_circles": "present",
            "spots": [{"location": "x=20, y=220", "size": 6}]
        },
        "nail_analysis": {
            "color": {"RGB": [70, 75, 80], "Hex": "#464c50"},
            "texture": {"edge_intensity": 0.6},
            "shape": {"shapes_detected": ["Irregular"]}
        },
        "outputs": {
            "disease": "eczema",
            "dosha_analysis": {
                "vata": "highly imbalanced",
                "pitta": "slightly imbalanced",
                "kapha": "balanced"
            },
            "observations": [
                "Rough skin texture indicates Vata imbalance.",
                "Irregular nail shapes suggest stress-related Vata issues."
            ],
            "recommendations": {
                "diet": ["Include warm, moist foods like soups.", "Avoid raw and cold foods."],
                "lifestyle": ["Apply coconut oil to affected areas.", "Practice grounding exercises like meditation."],
                "herbal_remedies": ["Neem for skin detoxification.", "Brahmi to calm the mind."]
            }
        }
    },
     {
        "skin_analysis": {
            "tone": "medium",
            "texture": "dry",
            "dark_circles": "absent",
            "spots": [{"location": "x=18, y=190", "size": 2}]
        },
        "nail_analysis": {
            "color": {"RGB": [88, 92, 95], "Hex": "#585c5f"},
            "texture": {"edge_intensity": 0.2},
            "shape": {"shapes_detected": ["Round"]}
        },
        "outputs": {
            "disease": "psoriasis",
            "dosha_analysis": {
                "vata": "imbalanced",
                "pitta": "highly imbalanced",
                "kapha": "balanced"
            },
            "observations": [
                "Dry skin and scaling indicate Pitta imbalance.",
                "Round nails suggest no Vata-related structural issues."
            ],
            "recommendations": {
                "diet": ["Include cooling foods like cucumber.", "Avoid spicy and acidic foods."],
                "lifestyle": ["Keep the skin moisturized.", "Avoid excessive sun exposure."],
                "herbal_remedies": ["Aloe vera gel for skin hydration.", "Manjistha to cool and purify the blood."]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "light",
            "texture": "smooth",
            "dark_circles": "present",
            "spots": [{"location": "x=12, y=205", "size": 5}]
        },
        "nail_analysis": {
            "color": {"RGB": [93, 98, 103], "Hex": "#5d6267"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Oval"]}
        },
        "outputs": {
            "disease": "hyperpigmentation",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "imbalanced",
                "kapha": "slightly imbalanced"
            },
            "observations": [
                "Pigmentation suggests an overactive Pitta.",
                "Nail shape indicates Kapha stagnation."
            ],
            "recommendations": {
                "diet": ["Include antioxidant-rich fruits.", "Reduce intake of oily and fried foods."],
                "lifestyle": ["Use natural sunscreen daily.", "Practice stress-relief activities like yoga."],
                "herbal_remedies": ["Turmeric for skin health.", "Neem for detoxification."]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "dark",
            "texture": "oily",
            "dark_circles": "absent",
            "spots": [{"location": "x=14, y=215", "size": 7}]
        },
        "nail_analysis": {
            "color": {"RGB": [65, 70, 75], "Hex": "#41464b"},
            "texture": {"edge_intensity": 0.5},
            "shape": {"shapes_detected": ["Irregular"]}
        },
        "outputs": {
            "disease": "acne",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "highly imbalanced",
                "kapha": "imbalanced"
            },
            "observations": [
                "Oily skin and acne suggest Pitta-Kapha imbalance.",
                "Irregular nails indicate stress-related Vata issues."
            ],
            "recommendations": {
                "diet": ["Eat light, cooling foods like salads.", "Avoid dairy and processed sugars."],
                "lifestyle": ["Wash the face twice daily with a mild cleanser.", "Avoid excessive use of makeup."],
                "herbal_remedies": ["Triphala for detoxification.", "Tulsi capsules for balancing Kapha."]
            }
        }
    },
    {
    "skin_analysis": {
        "tone": "light",
        "texture": "smooth",
        "dark_circles": "present",
        "spots": [{"location": "x=10, y=200", "size": 4}]
    },
    "nail_analysis": {
        "color": {"RGB": [90, 100, 110], "Hex": "#5a646e"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Round"]}
    },
    "outputs": {
        "disease": "vitiligo",
        "dosha_analysis": {
            "vata": "mildly imbalanced",
            "pitta": "balanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Skin pigmentation changes indicate mild Kapha imbalance.",
            "Round nail shape suggests a balanced Vata condition."
        ],
        "recommendations": {
            "diet": ["Increase intake of leafy greens.", "Avoid fried and spicy foods."],
            "lifestyle": ["Practice yoga to enhance circulation.", "Use sunscreen daily."],
            "herbal_remedies": ["Neem capsules for skin health.", "Ashwagandha for stress relief."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "oily",
        "dark_circles": "absent",
        "spots": [{"location": "x=30, y=150", "size": 6}]
    },
    "nail_analysis": {
        "color": {"RGB": [120, 90, 80], "Hex": "#785a50"},
        "texture": {"edge_intensity": 0.7},
        "shape": {"shapes_detected": ["Oval"]}
    },
    "outputs": {
        "disease": "acne",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "balanced"
        },
        "observations": [
            "Oily skin texture indicates a mild Pitta imbalance.",
            "Oval nail shape suggests proper Kapha balance."
        ],
        "recommendations": {
            "diet": ["Include cooling foods like cucumber and watermelon.", "Avoid spicy and oily foods."],
            "lifestyle": ["Wash face with mild cleansers twice a day.", "Avoid excessive sun exposure."],
            "herbal_remedies": ["Neem paste for acne spots.", "Turmeric supplement for inflammation."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "dry",
        "dark_circles": "present",
        "spots": [{"location": "x=50, y=220", "size": 8}]
    },
    "nail_analysis": {
        "color": {"RGB": [80, 60, 40], "Hex": "#503c28"},
        "texture": {"edge_intensity": 0.2},
        "shape": {"shapes_detected": ["Square"]}
    },
    "outputs": {
        "disease": "eczema",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "mildly imbalanced"
        },
        "observations": [
            "Dry skin texture and spots indicate severe Vata imbalance.",
            "Square nail shape indicates mild Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Eat warm, moist foods like soups and stews.", "Avoid cold and raw foods."],
            "lifestyle": ["Apply moisturizers to affected areas.", "Use natural oils like coconut oil for hydration."],
            "herbal_remedies": ["Aloe vera gel for skin irritation.", "Ashwagandha to reduce stress."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "rough",
        "dark_circles": "present",
        "spots": [{"location": "x=70, y=180", "size": 5}]
    },
    "nail_analysis": {
        "color": {"RGB": [150, 130, 90], "Hex": "#96825a"},
        "texture": {"edge_intensity": 0.6},
        "shape": {"shapes_detected": ["Almond"]}
    },
    "outputs": {
        "disease": "psoriasis",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "mildly imbalanced",
            "kapha": "balanced"
        },
        "observations": [
            "Rough skin texture and spots indicate severe Vata imbalance.",
            "Almond nail shape suggests a balanced Kapha condition."
        ],
        "recommendations": {
            "diet": ["Consume anti-inflammatory foods like flaxseeds and fish.", "Avoid processed foods and sugar."],
            "lifestyle": ["Practice meditation to reduce stress.", "Apply hydrating creams to affected areas."],
            "herbal_remedies": ["Neem oil for skin irritation.", "Turmeric capsules for inflammation."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "smooth",
        "dark_circles": "absent",
        "spots": [{"location": "x=20, y=300", "size": 3}]
    },
    "nail_analysis": {
        "color": {"RGB": [70, 80, 90], "Hex": "#46505a"},
        "texture": {"edge_intensity": 0.4},
        "shape": {"shapes_detected": ["Stiletto"]}
    },
    "outputs": {
        "disease": "melasma",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "severely imbalanced",
            "kapha": "mildly imbalanced"
        },
        "observations": [
            "Smooth skin texture and pigmentation indicate severe Pitta imbalance.",
            "Stiletto nail shape suggests mild Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Include foods rich in antioxidants like berries.", "Avoid alcohol and caffeine."],
            "lifestyle": ["Use sunscreen daily to prevent pigmentation.", "Avoid excessive heat exposure."],
            "herbal_remedies": ["Aloe vera gel for pigmentation.", "Licorice extract for lightening spots."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "smooth",
        "dark_circles": "present",
        "spots": [{"location": "x=10, y=200", "size": 4}]
    },
    "nail_analysis": {
        "color": {"RGB": [90, 100, 110], "Hex": "#5a646e"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Round"]}
    },
    "outputs": {
        "disease": "vitiligo",
        "dosha_analysis": {
            "vata": "mildly imbalanced",
            "pitta": "balanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Skin pigmentation changes indicate mild Kapha imbalance.",
            "Round nail shape suggests a balanced Vata condition."
        ],
        "recommendations": {
            "diet": ["Increase intake of leafy greens.", "Avoid fried and spicy foods."],
            "lifestyle": ["Practice yoga to enhance circulation.", "Use sunscreen daily."],
            "herbal_remedies": ["Neem capsules for skin health.", "Ashwagandha for stress relief."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "oily",
        "dark_circles": "absent",
        "spots": [{"location": "x=30, y=150", "size": 6}]
    },
    "nail_analysis": {
        "color": {"RGB": [120, 90, 80], "Hex": "#785a50"},
        "texture": {"edge_intensity": 0.7},
        "shape": {"shapes_detected": ["Oval"]}
    },
    "outputs": {
        "disease": "acne",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "balanced"
        },
        "observations": [
            "Oily skin texture indicates a mild Pitta imbalance.",
            "Oval nail shape suggests proper Kapha balance."
        ],
        "recommendations": {
            "diet": ["Include cooling foods like cucumber and watermelon.", "Avoid spicy and oily foods."],
            "lifestyle": ["Wash face with mild cleansers twice a day.", "Avoid excessive sun exposure."],
            "herbal_remedies": ["Neem paste for acne spots.", "Turmeric supplement for inflammation."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "dry",
        "dark_circles": "present",
        "spots": [{"location": "x=50, y=220", "size": 8}]
    },
    "nail_analysis": {
        "color": {"RGB": [80, 60, 40], "Hex": "#503c28"},
        "texture": {"edge_intensity": 0.2},
        "shape": {"shapes_detected": ["Square"]}
    },
    "outputs": {
        "disease": "eczema",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "mildly imbalanced"
        },
        "observations": [
            "Dry skin texture and spots indicate severe Vata imbalance.",
            "Square nail shape indicates mild Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Eat warm, moist foods like soups and stews.", "Avoid cold and raw foods."],
            "lifestyle": ["Apply moisturizers to affected areas.", "Use natural oils like coconut oil for hydration."],
            "herbal_remedies": ["Aloe vera gel for skin irritation.", "Ashwagandha to reduce stress."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "rough",
        "dark_circles": "present",
        "spots": [{"location": "x=70, y=180", "size": 5}]
    },
    "nail_analysis": {
        "color": {"RGB": [150, 130, 90], "Hex": "#96825a"},
        "texture": {"edge_intensity": 0.6},
        "shape": {"shapes_detected": ["Almond"]}
    },
    "outputs": {
        "disease": "psoriasis",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "mildly imbalanced",
            "kapha": "balanced"
        },
        "observations": [
            "Rough skin texture and spots indicate severe Vata imbalance.",
            "Almond nail shape suggests a balanced Kapha condition."
        ],
        "recommendations": {
            "diet": ["Consume anti-inflammatory foods like flaxseeds and fish.", "Avoid processed foods and sugar."],
            "lifestyle": ["Practice meditation to reduce stress.", "Apply hydrating creams to affected areas."],
            "herbal_remedies": ["Neem oil for skin irritation.", "Turmeric capsules for inflammation."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "smooth",
        "dark_circles": "absent",
        "spots": [{"location": "x=20, y=300", "size": 3}]
    },
    "nail_analysis": {
        "color": {"RGB": [70, 80, 90], "Hex": "#46505a"},
        "texture": {"edge_intensity": 0.4},
        "shape": {"shapes_detected": ["Stiletto"]}
    },
    "outputs": {
        "disease": "melasma",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "severely imbalanced",
            "kapha": "mildly imbalanced"
        },
        "observations": [
            "Smooth skin texture and pigmentation indicate severe Pitta imbalance.",
            "Stiletto nail shape suggests mild Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Include foods rich in antioxidants like berries.", "Avoid alcohol and caffeine."],
            "lifestyle": ["Use sunscreen daily to prevent pigmentation.", "Avoid excessive heat exposure."],
            "herbal_remedies": ["Aloe vera gel for pigmentation.", "Licorice extract for lightening spots."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "dry",
        "dark_circles": "present",
        "spots": [{"location": "x=40, y=180", "size": 6}]
    },
    "nail_analysis": {
        "color": {"RGB": [200, 180, 150], "Hex": "#c8b496"},
        "texture": {"edge_intensity": 0.1},
        "shape": {"shapes_detected": ["Oval"]}
    },
    "outputs": {
        "disease": "rosacea",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "mildly imbalanced",
            "kapha": "balanced"
        },
        "observations": [
            "Dry skin texture indicates a severe Vata imbalance.",
            "Oval nail shape suggests Kapha balance."
        ],
        "recommendations": {
            "diet": ["Consume hydrating and cooling foods like cucumber and yogurt.", "Avoid spicy and fried foods."],
            "lifestyle": ["Keep skin moisturized.", "Avoid exposure to extreme heat or cold."],
            "herbal_remedies": ["Aloe vera gel for skin hydration.", "Chamomile tea to reduce redness."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "oily",
        "dark_circles": "absent",
        "spots": [{"location": "x=60, y=150", "size": 7}]
    },
    "nail_analysis": {
        "color": {"RGB": [160, 140, 100], "Hex": "#a08c64"},
        "texture": {"edge_intensity": 0.5},
        "shape": {"shapes_detected": ["Square"]}
    },
    "outputs": {
        "disease": "seborrheic dermatitis",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "severely imbalanced",
            "kapha": "mildly imbalanced"
        },
        "observations": [
            "Oily skin and large spots indicate severe Pitta imbalance.",
            "Square nail shape suggests mild Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Reduce intake of dairy and fried foods.", "Consume more leafy greens and fruits."],
            "lifestyle": ["Wash scalp and skin frequently with mild cleansers.", "Avoid stress and maintain proper hydration."],
            "herbal_remedies": ["Neem oil for scalp health.", "Turmeric paste for inflammation."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "rough",
        "dark_circles": "present",
        "spots": [{"location": "x=10, y=300", "size": 5}]
    },
    "nail_analysis": {
        "color": {"RGB": [80, 70, 60], "Hex": "#50463c"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Almond"]}
    },
    "outputs": {
        "disease": "lichen planus",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "mildly imbalanced"
        },
        "observations": [
            "Rough skin texture and dark spots indicate severe Vata imbalance.",
            "Almond nail shape suggests a balanced Kapha condition."
        ],
        "recommendations": {
            "diet": ["Consume warm, cooked foods like soups and stews.", "Avoid cold and dry foods."],
            "lifestyle": ["Apply moisturizing creams regularly.", "Avoid excessive exposure to sunlight."],
            "herbal_remedies": ["Aloe vera for hydration.", "Ashwagandha for stress management."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "oily",
        "dark_circles": "present",
        "spots": [{"location": "x=15, y=220", "size": 4}]
    },
    "nail_analysis": {
        "color": {"RGB": [200, 170, 130], "Hex": "#c8aa82"},
        "texture": {"edge_intensity": 0.6},
        "shape": {"shapes_detected": ["Oval"]}
    },
    "outputs": {
        "disease": "acne vulgaris",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "balanced"
        },
        "observations": [
            "Oily skin and small spots suggest mild Pitta imbalance.",
            "Oval nail shape suggests Kapha balance."
        ],
        "recommendations": {
            "diet": ["Include cooling foods like cucumbers and watermelon.", "Avoid fried and spicy foods."],
            "lifestyle": ["Wash face twice daily with a mild cleanser.", "Avoid touching or picking at acne spots."],
            "herbal_remedies": ["Neem paste for acne treatment.", "Turmeric capsules for reducing inflammation."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "rough",
        "dark_circles": "absent",
        "spots": [{"location": "x=25, y=300", "size": 6}]
    },
    "nail_analysis": {
        "color": {"RGB": [100, 80, 60], "Hex": "#64503c"},
        "texture": {"edge_intensity": 0.2},
        "shape": {"shapes_detected": ["Square"]}
    },
    "outputs": {
        "disease": "eczema",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "mildly imbalanced"
        },
        "observations": [
            "Rough skin and spots indicate severe Vata imbalance.",
            "Square nail shape suggests mild Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Consume warm, cooked foods.", "Avoid raw and cold foods."],
            "lifestyle": ["Apply moisturizers frequently.", "Avoid harsh soaps or detergents."],
            "herbal_remedies": ["Aloe vera gel for hydration.", "Chamomile tea for inflammation."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "smooth",
        "dark_circles": "present",
        "spots": [{"location": "x=15, y=180", "size": 4}]
    },
    "nail_analysis": {
        "color": {"RGB": [160, 140, 120], "Hex": "#a08c78"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Round"]}
    },
    "outputs": {
        "disease": "iron deficiency anemia",
        "dosha_analysis": {
            "vata": "mildly imbalanced",
            "pitta": "balanced",
            "kapha": "balanced"
        },
        "observations": [
            "Dark circles indicate possible anemia due to Vata imbalance.",
            "Round nail shape suggests balanced Kapha and Pitta doshas."
        ],
        "recommendations": {
            "diet": ["Increase iron-rich foods like spinach, lentils, and red meat.", "Avoid caffeine during meals."],
            "lifestyle": ["Incorporate gentle exercise like walking.", "Ensure 7-8 hours of sleep daily."],
            "herbal_remedies": ["Amla for improving iron absorption.", "Ashwagandha to improve energy levels."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "rough",
        "dark_circles": "absent",
        "spots": [{"location": "x=60, y=220", "size": 8}]
    },
    "nail_analysis": {
        "color": {"RGB": [100, 80, 60], "Hex": "#64503c"},
        "texture": {"edge_intensity": 0.4},
        "shape": {"shapes_detected": ["Square"]}
    },
    "outputs": {
        "disease": "eczema",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "mildly imbalanced"
        },
        "observations": [
            "Rough skin texture indicates severe Vata imbalance.",
            "Square nail shape suggests mild Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Consume warm, moist foods like soups.", "Avoid raw and cold foods."],
            "lifestyle": ["Apply moisturizing creams frequently.", "Avoid exposure to irritants like harsh soaps."],
            "herbal_remedies": ["Neem oil for skin irritation.", "Chamomile tea to reduce inflammation."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "oily",
        "dark_circles": "present",
        "spots": [{"location": "x=45, y=150", "size": 5}]
    },
    "nail_analysis": {
        "color": {"RGB": [220, 200, 180], "Hex": "#dcc8b4"},
        "texture": {"edge_intensity": 0.7},
        "shape": {"shapes_detected": ["Oval"]}
    },
    "outputs": {
        "disease": "acne",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "balanced"
        },
        "observations": [
            "Oily skin texture indicates mild Pitta imbalance.",
            "Oval nail shape suggests Kapha balance."
        ],
        "recommendations": {
            "diet": ["Include cooling foods like cucumber and watermelon.", "Avoid spicy and greasy foods."],
            "lifestyle": ["Wash face with a mild cleanser twice daily.", "Avoid touching or picking at acne spots."],
            "herbal_remedies": ["Neem paste for acne treatment.", "Turmeric capsules for reducing inflammation."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "dry",
        "dark_circles": "present",
        "spots": [{"location": "x=25, y=180", "size": 3}]
    },
    "nail_analysis": {
        "color": {"RGB": [150, 130, 100], "Hex": "#968264"},
        "texture": {"edge_intensity": 0.2},
        "shape": {"shapes_detected": ["Square"]}
    },
    "outputs": {
        "disease": "psoriasis",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "mildly imbalanced"
        },
        "observations": [
            "Dry skin and spots indicate severe Vata imbalance.",
            "Square nail shape suggests mild Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Consume anti-inflammatory foods like flaxseeds and fish.", "Avoid processed foods and excessive sugar."],
            "lifestyle": ["Apply hydrating creams to affected areas.", "Avoid stress through meditation or yoga."],
            "herbal_remedies": ["Neem oil for skin hydration.", "Turmeric capsules for reducing inflammation."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "smooth",
        "dark_circles": "absent",
        "spots": [{"location": "x=15, y=260", "size": 6}]
    },
    "nail_analysis": {
        "color": {"RGB": [50, 60, 70], "Hex": "#323c46"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Almond"]}
    },
    "outputs": {
        "disease": "melasma",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "severely imbalanced",
            "kapha": "mildly imbalanced"
        },
        "observations": [
            "Smooth skin pigmentation suggests severe Pitta imbalance.",
            "Almond nail shape indicates mild Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Consume antioxidant-rich foods like berries and green tea.", "Avoid caffeine and alcohol."],
            "lifestyle": ["Use sunscreen daily to reduce pigmentation.", "Avoid direct sunlight exposure."],
            "herbal_remedies": ["Aloe vera gel for soothing pigmentation.", "Licorice extract for pigmentation reduction."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "rough",
        "dark_circles": "present",
        "spots": [{"location": "x=20, y=220", "size": 4}]
    },
    "nail_analysis": {
        "color": {"RGB": [220, 200, 180], "Hex": "#dcc8b4"},
        "texture": {"edge_intensity": 0.6},
        "shape": {"shapes_detected": ["Round"]}
    },
    "outputs": {
        "disease": "keratosis pilaris",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "balanced"
        },
        "observations": [
            "Rough skin texture and small spots suggest severe Vata imbalance.",
            "Round nail shape indicates Kapha balance."
        ],
        "recommendations": {
            "diet": ["Consume warm, cooked foods to balance Vata.", "Avoid cold and raw foods."],
            "lifestyle": ["Gently exfoliate affected areas.", "Apply moisturizing creams regularly."],
            "herbal_remedies": ["Aloe vera gel for hydration.", "Ashwagandha to reduce dryness."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "oily",
        "dark_circles": "present",
        "spots": [{"location": "x=25, y=120", "size": 5}]
    },
    "nail_analysis": {
        "color": {"RGB": [230, 210, 190], "Hex": "#e6d2be"},
        "texture": {"edge_intensity": 0.7},
        "shape": {"shapes_detected": ["Round"]}
    },
    "outputs": {
        "disease": "acne",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Oily skin and spots suggest mild Pitta imbalance.",
            "Round nail shape indicates a slightly imbalanced Kapha condition."
        ],
        "recommendations": {
            "diet": ["Consume more cooling foods like cucumber and watermelon.", "Avoid spicy and fried foods."],
            "lifestyle": ["Wash your face twice daily with a mild cleanser.", "Avoid touching your face frequently."],
            "herbal_remedies": ["Neem paste for acne treatment.", "Turmeric capsules for inflammation control."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "dry",
        "dark_circles": "absent",
        "spots": [{"location": "x=35, y=200", "size": 4}]
    },
    "nail_analysis": {
        "color": {"RGB": [160, 130, 100], "Hex": "#a08264"},
        "texture": {"edge_intensity": 0.2},
        "shape": {"shapes_detected": ["Square"]}
    },
    "outputs": {
        "disease": "eczema",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "mildly imbalanced"
        },
        "observations": [
            "Dry skin texture indicates severe Vata imbalance.",
            "Square nail shape suggests mild Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Consume warm, moist foods like soups and stews.", "Avoid cold and raw foods."],
            "lifestyle": ["Apply rich moisturizers frequently.", "Avoid exposure to irritants like harsh detergents."],
            "herbal_remedies": ["Aloe vera gel for hydration.", "Chamomile tea to reduce redness."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "rough",
        "dark_circles": "present",
        "spots": [{"location": "x=15, y=250", "size": 6}]
    },
    "nail_analysis": {
        "color": {"RGB": [100, 80, 70], "Hex": "#645046"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Almond"]}
    },
    "outputs": {
        "disease": "keratosis pilaris",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "balanced"
        },
        "observations": [
            "Rough skin texture indicates severe Vata imbalance.",
            "Almond nail shape suggests Kapha balance."
        ],
        "recommendations": {
            "diet": ["Consume warm, cooked foods to balance Vata.", "Avoid cold and dry foods."],
            "lifestyle": ["Exfoliate gently with a scrub.", "Apply moisturizing creams daily."],
            "herbal_remedies": ["Aloe vera gel for hydration.", "Ashwagandha powder to reduce dryness."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "smooth",
        "dark_circles": "present",
        "spots": [{"location": "x=50, y=180", "size": 3}]
    },
    "nail_analysis": {
        "color": {"RGB": [190, 170, 150], "Hex": "#bea996"},
        "texture": {"edge_intensity": 0.6},
        "shape": {"shapes_detected": ["Oval"]}
    },
    "outputs": {
        "disease": "melasma",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "severely imbalanced",
            "kapha": "balanced"
        },
        "observations": [
            "Smooth skin pigmentation spots suggest severe Pitta imbalance.",
            "Oval nail shape indicates Kapha balance."
        ],
        "recommendations": {
            "diet": ["Consume antioxidant-rich foods like blueberries and spinach.", "Avoid spicy and greasy foods."],
            "lifestyle": ["Use sunscreen daily to protect against pigmentation.", "Avoid prolonged sun exposure."],
            "herbal_remedies": ["Licorice extract for pigmentation lightening.", "Aloe vera gel for soothing the skin."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "oily",
        "dark_circles": "absent",
        "spots": [{"location": "x=30, y=220", "size": 7}]
    },
    "nail_analysis": {
        "color": {"RGB": [170, 150, 120], "Hex": "#aa9678"},
        "texture": {"edge_intensity": 0.5},
        "shape": {"shapes_detected": ["Round"]}
    },
    "outputs": {
        "disease": "seborrheic dermatitis",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "balanced"
        },
        "observations": [
            "Oily skin and large spots suggest mild Pitta imbalance.",
            "Round nail shape indicates balanced Kapha condition."
        ],
        "recommendations": {
            "diet": ["Avoid fried foods and include more greens in your diet.", "Stay hydrated to reduce oil production."],
            "lifestyle": ["Wash your scalp and face often with mild cleansers.", "Use anti-fungal shampoos if necessary."],
            "herbal_remedies": ["Neem oil for scalp irritation.", "Turmeric paste for reducing inflammation."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "smooth",
        "dark_circles": "present",
        "spots": [{"location": "x=20, y=140", "size": 4}]
    },
    "nail_analysis": {
        "color": {"RGB": [60, 50, 40], "Hex": "#3c3228"},
        "texture": {"edge_intensity": 0.2},
        "shape": {"shapes_detected": ["Square"]}
    },
    "outputs": {
        "disease": "lichen planus",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "mildly imbalanced"
        },
        "observations": [
            "Smooth skin with pigmentation indicates severe Vata imbalance.",
            "Square nail shape suggests mild Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Consume warm, nourishing foods.", "Avoid caffeine and alcohol."],
            "lifestyle": ["Apply natural oils like coconut oil to keep the skin hydrated.", "Practice stress-relieving exercises like yoga."],
            "herbal_remedies": ["Aloe vera gel for hydration.", "Ashwagandha for managing stress."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "oily",
        "dark_circles": "present",
        "spots": [{"location": "x=20, y=150", "size": 6}]
    },
    "nail_analysis": {
        "color": {"RGB": [80, 60, 50], "Hex": "#503c32"},
        "texture": {"edge_intensity": 0.5},
        "shape": {"shapes_detected": ["Round"]}
    },
    "outputs": {
        "disease": "acne",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Oily skin and large spots indicate mild Pitta imbalance.",
            "Round nail shape reflects Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Consume cooling foods like cucumbers and melons.", "Avoid fried and spicy foods."],
            "lifestyle": ["Wash face with a gentle cleanser twice daily.", "Reduce exposure to humid environments."],
            "herbal_remedies": ["Neem paste for acne treatment.", "Turmeric for reducing inflammation."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "smooth",
        "dark_circles": "absent",
        "spots": [{"location": "x=35, y=200", "size": 4}]
    },
    "nail_analysis": {
        "color": {"RGB": [220, 200, 180], "Hex": "#dcc8b4"},
        "texture": {"edge_intensity": 0.2},
        "shape": {"shapes_detected": ["Square"]}
    },
    "outputs": {
        "disease": "vitiligo",
        "dosha_analysis": {
            "vata": "mildly imbalanced",
            "pitta": "balanced",
            "kapha": "balanced"
        },
        "observations": [
            "Skin depigmentation indicates mild Vata imbalance.",
            "Square nail shape suggests balanced doshas."
        ],
        "recommendations": {
            "diet": ["Increase intake of leafy greens and whole grains.", "Avoid processed and sugary foods."],
            "lifestyle": ["Use sunscreen daily to protect depigmented areas.", "Maintain a consistent sleep schedule."],
            "herbal_remedies": ["Neem capsules for skin health.", "Ashwagandha for reducing stress."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "dry",
        "dark_circles": "present",
        "spots": [{"location": "x=40, y=250", "size": 5}]
    },
    "nail_analysis": {
        "color": {"RGB": [180, 160, 140], "Hex": "#b49c8c"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Oval"]}
    },
    "outputs": {
        "disease": "eczema",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "mildly imbalanced"
        },
        "observations": [
            "Dry skin texture and spots suggest severe Vata imbalance.",
            "Oval nail shape reflects mild Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Consume warm soups and stews.", "Avoid cold and raw foods."],
            "lifestyle": ["Apply moisturizing creams to affected areas.", "Avoid exposure to harsh soaps or detergents."],
            "herbal_remedies": ["Aloe vera gel for hydration.", "Ashwagandha to reduce dryness."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "rough",
        "dark_circles": "present",
        "spots": [{"location": "x=30, y=300", "size": 7}]
    },
    "nail_analysis": {
        "color": {"RGB": [60, 50, 40], "Hex": "#3c3228"},
        "texture": {"edge_intensity": 0.4},
        "shape": {"shapes_detected": ["Almond"]}
    },
    "outputs": {
        "disease": "lichen planus",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "balanced"
        },
        "observations": [
            "Rough skin and dark spots indicate severe Vata imbalance.",
            "Almond nail shape reflects balanced Kapha and Pitta."
        ],
        "recommendations": {
            "diet": ["Consume warm, nourishing foods like stews.", "Avoid caffeine and alcohol."],
            "lifestyle": ["Apply natural oils like coconut oil to hydrate skin.", "Engage in stress-relieving activities like yoga."],
            "herbal_remedies": ["Neem oil for skin irritation.", "Chamomile tea to reduce inflammation."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "smooth",
        "dark_circles": "absent",
        "spots": [{"location": "x=20, y=180", "size": 3}]
    },
    "nail_analysis": {
        "color": {"RGB": [200, 170, 140], "Hex": "#c8aa8c"},
        "texture": {"edge_intensity": 0.2},
        "shape": {"shapes_detected": ["Round"]}
    },
    "outputs": {
        "disease": "rosacea",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "balanced"
        },
        "observations": [
            "Smooth skin with redness suggests mild Pitta imbalance.",
            "Round nail shape reflects balanced Kapha and Vata doshas."
        ],
        "recommendations": {
            "diet": ["Consume hydrating foods like cucumber and yogurt.", "Avoid hot and spicy foods."],
            "lifestyle": ["Use gentle cleansers to avoid skin irritation.", "Protect skin from extreme heat or cold."],
            "herbal_remedies": ["Chamomile tea to soothe redness.", "Aloe vera gel for hydration."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "oily",
        "dark_circles": "present",
        "spots": [{"location": "x=35, y=150", "size": 6}]
    },
    "nail_analysis": {
        "color": {"RGB": [150, 130, 110], "Hex": "#96826e"},
        "texture": {"edge_intensity": 0.5},
        "shape": {"shapes_detected": ["Square"]}
    },
    "outputs": {
        "disease": "seborrheic dermatitis",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Oily skin and large spots suggest mild Pitta imbalance.",
            "Square nail shape reflects slightly imbalanced Kapha."
        ],
        "recommendations": {
            "diet": ["Reduce fried and greasy foods.", "Consume more fresh vegetables and fruits."],
            "lifestyle": ["Wash scalp and face with mild cleansers.", "Avoid scratching or irritating affected areas."],
            "herbal_remedies": ["Neem oil for scalp health.", "Turmeric paste to reduce inflammation."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "oily",
        "dark_circles": "present",
        "spots": [{"location": "x=20, y=300", "size": 10}]
    },
    "nail_analysis": {
        "color": {"RGB": [100, 80, 60], "Hex": "#64503c"},
        "texture": {"edge_intensity": 0.8},
        "shape": {"shapes_detected": ["Round"]}
    },
    "outputs": {
        "disease": "hyperpigmentation",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "severely imbalanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Oily skin texture and dark spots suggest Pitta imbalance.",
            "Round nail shape indicates mild Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Consume antioxidant-rich foods like berries and green tea.", "Avoid spicy, fried, and greasy foods."],
            "lifestyle": ["Use sunscreen regularly to prevent further pigmentation.", "Avoid prolonged exposure to sunlight."],
            "herbal_remedies": ["Licorice root for skin lightening.", "Neem capsules for detoxification."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "rough",
        "dark_circles": "absent",
        "spots": [{"location": "x=50, y=180", "size": 3}]
    },
    "nail_analysis": {
        "color": {"RGB": [200, 160, 140], "Hex": "#c8a08c"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Oval"]}
    },
    "outputs": {
        "disease": "keratosis pilaris",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "balanced"
        },
        "observations": [
            "Rough skin texture indicates severe Vata imbalance.",
            "Oval nail shape suggests balanced Kapha condition."
        ],
        "recommendations": {
            "diet": ["Consume warm, cooked foods like soups and lentils.", "Avoid cold and dry foods."],
            "lifestyle": ["Exfoliate gently with natural scrubs.", "Apply moisturizers or coconut oil daily."],
            "herbal_remedies": ["Aloe vera gel for hydration.", "Ashwagandha to reduce dryness."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "smooth",
        "dark_circles": "present",
        "spots": [{"location": "x=15, y=250", "size": 6}]
    },
    "nail_analysis": {
        "color": {"RGB": [140, 130, 120], "Hex": "#8c8278"},
        "texture": {"edge_intensity": 0.5},
        "shape": {"shapes_detected": ["Square"]}
    },
    "outputs": {
        "disease": "psoriasis",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "mildly imbalanced",
            "kapha": "balanced"
        },
        "observations": [
            "Smooth skin with scaling indicates severe Vata imbalance.",
            "Square nail shape suggests balanced Kapha condition."
        ],
        "recommendations": {
            "diet": ["Add anti-inflammatory foods like flaxseeds and omega-3-rich fish.", "Avoid processed and sugary foods."],
            "lifestyle": ["Apply hydrating creams frequently.", "Practice meditation to reduce stress."],
            "herbal_remedies": ["Neem oil for skin irritation.", "Turmeric capsules for inflammation reduction."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "oily",
        "dark_circles": "absent",
        "spots": [{"location": "x=40, y=190", "size": 8}]
    },
    "nail_analysis": {
        "color": {"RGB": [90, 60, 50], "Hex": "#5a3c32"},
        "texture": {"edge_intensity": 0.7},
        "shape": {"shapes_detected": ["Round"]}
    },
    "outputs": {
        "disease": "acne vulgaris",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Oily skin and large spots suggest mild Pitta imbalance.",
            "Round nail shape indicates Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Include more cooling foods like cucumber and watermelon.", "Avoid dairy and fried foods."],
            "lifestyle": ["Wash face with a mild cleanser twice daily.", "Avoid touching or picking at acne spots."],
            "herbal_remedies": ["Neem paste for acne treatment.", "Tulsi tea for detoxification."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "dry",
        "dark_circles": "present",
        "spots": [{"location": "x=50, y=230", "size": 4}]
    },
    "nail_analysis": {
        "color": {"RGB": [170, 140, 100], "Hex": "#aa8c64"},
        "texture": {"edge_intensity": 0.2},
        "shape": {"shapes_detected": ["Stiletto"]}
    },
    "outputs": {
        "disease": "eczema",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "mildly imbalanced"
        },
        "observations": [
            "Dry skin with scaling and spots suggests severe Vata imbalance.",
            "Stiletto nail shape indicates mild Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Eat warm, moist foods like soups and stews.", "Avoid raw and cold foods."],
            "lifestyle": ["Apply natural moisturizers like shea butter.", "Avoid irritants like harsh soaps."],
            "herbal_remedies": ["Chamomile tea for calming inflammation.", "Aloe vera gel for hydration."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "smooth",
        "dark_circles": "absent",
        "spots": [{"location": "x=20, y=140", "size": 2}]
    },
    "nail_analysis": {
        "color": {"RGB": [230, 210, 190], "Hex": "#e6d2be"},
        "texture": {"edge_intensity": 0.4},
        "shape": {"shapes_detected": ["Oval"]}
    },
    "outputs": {
        "disease": "melasma",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "severely imbalanced",
            "kapha": "balanced"
        },
        "observations": [
            "Smooth skin pigmentation suggests severe Pitta imbalance.",
            "Oval nail shape indicates balanced Kapha condition."
        ],
        "recommendations": {
            "diet": ["Consume antioxidant-rich foods like spinach and berries.", "Avoid spicy and greasy foods."],
            "lifestyle": ["Use sunscreen daily to prevent pigmentation.", "Avoid prolonged sun exposure."],
            "herbal_remedies": ["Licorice root for skin lightening.", "Aloe vera gel for soothing pigmentation."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "oily",
        "dark_circles": "present",
        "spots": [{"location": "x=30, y=150", "size": 7}]
    },
    "nail_analysis": {
        "color": {"RGB": [150, 120, 100], "Hex": "#967864"},
        "texture": {"edge_intensity": 0.6},
        "shape": {"shapes_detected": ["Oval"]}
    },
    "outputs": {
        "disease": "seborrheic dermatitis",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Oily skin texture and large spots suggest mild Pitta imbalance.",
            "Oval nail shape indicates slight Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Avoid fried and greasy foods.", "Consume more leafy greens and fruits."],
            "lifestyle": ["Wash scalp and skin with mild cleansers.", "Avoid stress and maintain hydration."],
            "herbal_remedies": ["Neem oil for scalp health.", "Turmeric paste for inflammation."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "rough",
        "dark_circles": "absent",
        "spots": [{"location": "x=40, y=200", "size": 5}]
    },
    "nail_analysis": {
        "color": {"RGB": [80, 70, 60], "Hex": "#50463c"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Square"]}
    },
    "outputs": {
        "disease": "lichen planus",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "mildly imbalanced"
        },
        "observations": [
            "Rough skin texture and spots indicate severe Vata imbalance.",
            "Square nail shape suggests mild Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Consume warm, nourishing foods like soups.", "Avoid cold and raw foods."],
            "lifestyle": ["Apply natural oils like coconut oil to hydrate skin.", "Practice stress-relieving exercises like yoga."],
            "herbal_remedies": ["Aloe vera gel for hydration.", "Ashwagandha for stress management."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "smooth",
        "dark_circles": "present",
        "spots": [{"location": "x=25, y=180", "size": 3}]
    },
    "nail_analysis": {
        "color": {"RGB": [200, 180, 160], "Hex": "#c8b4a0"},
        "texture": {"edge_intensity": 0.4},
        "shape": {"shapes_detected": ["Round"]}
    },
    "outputs": {
        "disease": "iron deficiency anemia",
        "dosha_analysis": {
            "vata": "mildly imbalanced",
            "pitta": "balanced",
            "kapha": "balanced"
        },
        "observations": [
            "Dark circles suggest possible anemia due to Vata imbalance.",
            "Round nail shape indicates balanced Kapha and Pitta doshas."
        ],
        "recommendations": {
            "diet": ["Increase intake of iron-rich foods like spinach and lentils.", "Avoid caffeine during meals."],
            "lifestyle": ["Incorporate gentle exercise like walking.", "Ensure 7-8 hours of sleep daily."],
            "herbal_remedies": ["Amla for improving iron absorption.", "Ashwagandha to improve energy levels."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "dry",
        "dark_circles": "absent",
        "spots": [{"location": "x=35, y=220", "size": 6}]
    },
    "nail_analysis": {
        "color": {"RGB": [170, 140, 110], "Hex": "#aa8c6e"},
        "texture": {"edge_intensity": 0.2},
        "shape": {"shapes_detected": ["Stiletto"]}
    },
    "outputs": {
        "disease": "eczema",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "mildly imbalanced"
        },
        "observations": [
            "Dry skin texture and spots indicate severe Vata imbalance.",
            "Stiletto nail shape suggests mild Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Consume warm, moist foods like soups and stews.", "Avoid raw and cold foods."],
            "lifestyle": ["Apply rich moisturizers frequently.", "Avoid exposure to irritants like harsh detergents."],
            "herbal_remedies": ["Aloe vera gel for hydration.", "Chamomile tea to reduce redness."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "oily",
        "dark_circles": "present",
        "spots": [{"location": "x=50, y=300", "size": 8}]
    },
    "nail_analysis": {
        "color": {"RGB": [90, 70, 50], "Hex": "#5a4632"},
        "texture": {"edge_intensity": 0.7},
        "shape": {"shapes_detected": ["Oval"]}
    },
    "outputs": {
        "disease": "acne vulgaris",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Oily skin and large spots suggest mild Pitta imbalance.",
            "Oval nail shape indicates slight Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Include more cooling foods like cucumber and watermelon.", "Avoid dairy and fried foods."],
            "lifestyle": ["Wash face with a mild cleanser twice daily.", "Avoid touching or picking at acne spots."],
            "herbal_remedies": ["Neem paste for acne treatment.", "Tulsi tea for detoxification."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "rough",
        "dark_circles": "absent",
        "spots": [{"location": "x=20, y=140", "size": 4}]
    },
    "nail_analysis": {
        "color": {"RGB": [220, 200, 180], "Hex": "#dcc8b4"},
        "texture": {"edge_intensity": 0.5},
        "shape": {"shapes_detected": ["Square"]}
    },
    "outputs": {
        "disease": "keratosis pilaris",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "balanced"
        },
        "observations": [
            "Rough skin texture indicates severe Vata imbalance.",
            "Square nail shape suggests balanced Kapha condition."
        ],
        "recommendations": {
            "diet": ["Consume warm, cooked foods like soups and lentils.", "Avoid cold and dry foods."],
            "lifestyle": ["Exfoliate gently with natural scrubs.", "Apply moisturizers or coconut oil daily."],
            "herbal_remedies": ["Aloe vera gel for hydration.", "Ashwagandha to reduce dryness."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "normal",
        "dark_circles": "present",
        "spots": [{"location": "x=10, y=100", "size": 2}]
    },
    "nail_analysis": {
        "color": {"RGB": [140, 120, 110], "Hex": "#8c786e"},
        "texture": {"edge_intensity": 0.4},
        "shape": {"shapes_detected": ["Oval"]}
    },
    "outputs": {
        "disease": "iron deficiency anemia",
        "dosha_analysis": {
            "vata": "mildly imbalanced",
            "pitta": "balanced",
            "kapha": "balanced"
        },
        "observations": [
            "Dark circles suggest mild anemia due to Vata imbalance.",
            "Oval nail shape reflects no major Kapha or Pitta imbalance."
        ],
        "recommendations": {
            "diet": ["Consume iron-rich foods like spinach, lentils, and red meat.", "Avoid caffeine during iron absorption."],
            "lifestyle": ["Engage in moderate exercise to improve circulation.", "Ensure regular sleep patterns for recovery."],
            "herbal_remedies": ["Amla for improving iron absorption.", "Ashwagandha to reduce fatigue."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "oily",
        "dark_circles": "absent",
        "spots": [{"location": "x=20, y=250", "size": 7}]
    },
    "nail_analysis": {
        "color": {"RGB": [120, 100, 80], "Hex": "#786450"},
        "texture": {"edge_intensity": 0.6},
        "shape": {"shapes_detected": ["Square"]}
    },
    "outputs": {
        "disease": "acne",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Oily skin texture and large spots suggest mild Pitta imbalance.",
            "Square nails indicate slight Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Consume cooling foods like cucumber and melons.", "Avoid fried and greasy foods."],
            "lifestyle": ["Wash face twice daily with a mild cleanser.", "Avoid touching affected areas."],
            "herbal_remedies": ["Neem paste for acne treatment.", "Turmeric capsules for reducing inflammation."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "dry",
        "dark_circles": "present",
        "spots": [{"location": "x=35, y=180", "size": 3}]
    },
    "nail_analysis": {
        "color": {"RGB": [200, 180, 160], "Hex": "#c8b4a0"},
        "texture": {"edge_intensity": 0.2},
        "shape": {"shapes_detected": ["Round"]}
    },
    "outputs": {
        "disease": "eczema",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "balanced"
        },
        "observations": [
            "Dry skin texture with small spots indicates severe Vata imbalance.",
            "Round nails indicate no significant Kapha or Pitta imbalance."
        ],
        "recommendations": {
            "diet": ["Eat warm, moist foods like soups and porridges.", "Avoid cold and raw foods."],
            "lifestyle": ["Apply moisturizers to affected areas.", "Avoid harsh soaps and detergents."],
            "herbal_remedies": ["Aloe vera gel for hydration.", "Chamomile tea for reducing inflammation."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "rough",
        "dark_circles": "absent",
        "spots": [{"location": "x=50, y=300", "size": 6}]
    },
    "nail_analysis": {
        "color": {"RGB": [170, 140, 130], "Hex": "#aa8c82"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Almond"]}
    },
    "outputs": {
        "disease": "keratosis pilaris",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "balanced"
        },
        "observations": [
            "Rough skin texture with small bumps suggests severe Vata imbalance.",
            "Almond nail shape indicates balanced Kapha and Pitta doshas."
        ],
        "recommendations": {
            "diet": ["Consume warm, cooked foods to balance Vata.", "Avoid cold and dry foods."],
            "lifestyle": ["Exfoliate gently with natural scrubs.", "Apply moisturizing creams regularly."],
            "herbal_remedies": ["Aloe vera gel for hydration.", "Ashwagandha powder for reducing dryness."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "smooth",
        "dark_circles": "present",
        "spots": [{"location": "x=15, y=200", "size": 8}]
    },
    "nail_analysis": {
        "color": {"RGB": [80, 70, 60], "Hex": "#50463c"},
        "texture": {"edge_intensity": 0.5},
        "shape": {"shapes_detected": ["Oval"]}
    },
    "outputs": {
        "disease": "melasma",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "severely imbalanced",
            "kapha": "balanced"
        },
        "observations": [
            "Smooth skin pigmentation suggests severe Pitta imbalance.",
            "Oval nails indicate balanced Kapha and Vata doshas."
        ],
        "recommendations": {
            "diet": ["Consume antioxidant-rich foods like spinach and berries.", "Avoid spicy and greasy foods."],
            "lifestyle": ["Use sunscreen daily to prevent further pigmentation.", "Avoid prolonged sun exposure."],
            "herbal_remedies": ["Licorice root for skin lightening.", "Aloe vera gel for soothing pigmentation."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "normal",
        "dark_circles": "absent",
        "spots": [{"location": "x=20, y=150", "size": 3}]
    },
    "nail_analysis": {
        "color": {"RGB": [220, 200, 180], "Hex": "#dcc8b4"},
        "texture": {"edge_intensity": 0.4},
        "shape": {"shapes_detected": ["Round"]}
    },
    "outputs": {
        "disease": "vitiligo",
        "dosha_analysis": {
            "vata": "mildly imbalanced",
            "pitta": "balanced",
            "kapha": "balanced"
        },
        "observations": [
            "Normal skin texture with depigmentation suggests mild Vata imbalance.",
            "Round nails indicate balance in Kapha and Pitta doshas."
        ],
        "recommendations": {
            "diet": ["Consume foods rich in antioxidants like oranges and green tea.", "Avoid processed and sugary foods."],
            "lifestyle": ["Apply sunscreen regularly to protect depigmented areas.", "Avoid stress through meditation."],
            "herbal_remedies": ["Neem capsules for skin health.", "Turmeric supplements to reduce inflammation."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "dry",
        "dark_circles": "present",
        "spots": [{"location": "x=30, y=120", "size": 4}]
    },
    "nail_analysis": {
        "color": {"RGB": [210, 180, 150], "Hex": "#d2b496"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Round"]}
    },
    "outputs": {
        "disease": "eczema",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Dry skin texture and visible irritation suggest severe Vata imbalance.",
            "Round nail shape reflects a slight Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Consume warm, nourishing foods like soups, ghee, and whole grains.", "Avoid cold, dry, and processed foods."],
            "lifestyle": ["Keep the skin hydrated by using sesame oil before bathing.", "Avoid harsh soaps or detergents."],
            "herbal_remedies": [
                "Manjistha for blood purification.",
                "Kumari (Aloe vera) gel to soothe irritated skin.",
                "Neem oil for topical application to reduce inflammation."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "oily",
        "dark_circles": "absent",
        "spots": [{"location": "x=40, y=200", "size": 5}]
    },
    "nail_analysis": {
        "color": {"RGB": [100, 90, 70], "Hex": "#645a46"},
        "texture": {"edge_intensity": 0.4},
        "shape": {"shapes_detected": ["Oval"]}
    },
    "outputs": {
        "disease": "acne",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Oily skin texture and visible acne suggest mild Pitta imbalance.",
            "Oval nail shape indicates slight Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Include cooling foods like cucumber, coconut water, and bitter gourd.", "Avoid fried, spicy, and fermented foods."],
            "lifestyle": ["Wash the face with a mild herbal cleanser twice daily.", "Avoid overusing cosmetic products."],
            "herbal_remedies": [
                "Turmeric and sandalwood paste for acne spots.",
                "Sariva (Indian sarsaparilla) to reduce heat and purify the blood.",
                "Triphala powder to detoxify the digestive system."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "rough",
        "dark_circles": "present",
        "spots": [{"location": "x=60, y=180", "size": 7}]
    },
    "nail_analysis": {
        "color": {"RGB": [140, 120, 100], "Hex": "#8c7864"},
        "texture": {"edge_intensity": 0.5},
        "shape": {"shapes_detected": ["Square"]}
    },
    "outputs": {
        "disease": "psoriasis",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "mildly imbalanced",
            "kapha": "balanced"
        },
        "observations": [
            "Rough, scaly skin along with redness suggests severe Vata and mild Pitta imbalance.",
            "Square nails indicate a balanced Kapha condition."
        ],
        "recommendations": {
            "diet": ["Consume anti-inflammatory foods like flaxseeds, turmeric milk, and green leafy vegetables.", "Avoid processed and sugary foods."],
            "lifestyle": ["Apply coconut oil or castor oil on affected areas to reduce dryness.", "Practice meditation or calming activities to manage stress."],
            "herbal_remedies": [
                "Guduchi (Tinospora cordifolia) for detoxification.",
                "Neem capsules for skin purification.",
                "Khadira (Acacia catechu) for reducing itching and inflammation."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "smooth",
        "dark_circles": "present",
        "spots": [{"location": "x=25, y=250", "size": 3}]
    },
    "nail_analysis": {
        "color": {"RGB": [160, 140, 120], "Hex": "#a08c78"},
        "texture": {"edge_intensity": 0.2},
        "shape": {"shapes_detected": ["Round"]}
    },
    "outputs": {
        "disease": "hyperpigmentation",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "severely imbalanced",
            "kapha": "balanced"
        },
        "observations": [
            "Smooth skin with pigmentation suggests severe Pitta imbalance.",
            "Round nail shape reflects a balanced Kapha and Vata condition."
        ],
        "recommendations": {
            "diet": ["Consume antioxidant-rich foods like amla, pomegranate, and carrots.", "Avoid caffeine and alcohol."],
            "lifestyle": ["Use sunscreen with herbal ingredients like turmeric daily.", "Avoid excessive sun exposure and heat."],
            "herbal_remedies": [
                "Chandan (sandalwood) paste for topical pigmentation treatment.",
                "Sariva (Indian sarsaparilla) to cool the body internally.",
                "Aloe vera gel to reduce inflammation and lighten spots."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "oily",
        "dark_circles": "absent",
        "spots": [{"location": "x=50, y=220", "size": 6}]
    },
    "nail_analysis": {
        "color": {"RGB": [140, 110, 90], "Hex": "#8c6e5a"},
        "texture": {"edge_intensity": 0.4},
        "shape": {"shapes_detected": ["Almond"]}
    },
    "outputs": {
        "disease": "seborrheic dermatitis",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "imbalanced"
        },
        "observations": [
            "Oily skin along with scaly patches indicates mild Pitta and Kapha imbalance.",
            "Almond nails reflect slightly imbalanced Kapha."
        ],
        "recommendations": {
            "diet": ["Include bitter greens like neem leaves and kale.", "Avoid dairy and fermented foods."],
            "lifestyle": ["Use herbal shampoos to cleanse the scalp.", "Wash hair regularly with lukewarm water."],
            "herbal_remedies": [
                "Bhringraj oil for scalp application.",
                "Triphala powder for detoxification.",
                "Neem oil to reduce itching and dandruff."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "rough",
        "dark_circles": "present",
        "spots": [{"location": "x=30, y=150", "size": 5}]
    },
    "nail_analysis": {
        "color": {"RGB": [140, 120, 100], "Hex": "#8c7864"},
        "texture": {"edge_intensity": 0.6},
        "shape": {"shapes_detected": ["Square"]}
    },
    "outputs": {
        "disease": "psoriasis",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "mildly imbalanced",
            "kapha": "balanced"
        },
        "observations": [
            "Rough skin texture and spots indicate severe Vata imbalance.",
            "Square nail shape suggests mild Pitta imbalance."
        ],
        "recommendations": {
            "diet": ["Consume anti-inflammatory foods like flaxseeds and fish.", "Avoid processed foods and sugar."],
            "lifestyle": ["Apply hydrating creams to affected areas.", "Practice stress-reducing activities like yoga."],
            "herbal_remedies": ["Neem oil for skin irritation.", "Turmeric capsules for inflammation."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "smooth",
        "dark_circles": "absent",
        "spots": [{"location": "x=20, y=300", "size": 3}]
    },
    "nail_analysis": {
        "color": {"RGB": [70, 60, 50], "Hex": "#463c32"},
        "texture": {"edge_intensity": 0.4},
        "shape": {"shapes_detected": ["Oval"]}
    },
    "outputs": {
        "disease": "melasma",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "severely imbalanced",
            "kapha": "mildly imbalanced"
        },
        "observations": [
            "Smooth skin pigmentation suggests severe Pitta imbalance.",
            "Oval nail shape indicates mild Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Consume antioxidant-rich foods like berries and spinach.", "Avoid spicy and greasy foods."],
            "lifestyle": ["Use sunscreen daily to prevent pigmentation.", "Avoid prolonged sun exposure."],
            "herbal_remedies": ["Licorice extract for pigmentation lightening.", "Aloe vera gel for soothing the skin."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "oily",
        "dark_circles": "present",
        "spots": [{"location": "x=15, y=180", "size": 6}]
    },
    "nail_analysis": {
        "color": {"RGB": [200, 180, 160], "Hex": "#c8b4a0"},
        "texture": {"edge_intensity": 0.7},
        "shape": {"shapes_detected": ["Round"]}
    },
    "outputs": {
        "disease": "acne vulgaris",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Oily skin and spots suggest mild Pitta imbalance.",
            "Round nail shape reflects slightly imbalanced Kapha."
        ],
        "recommendations": {
            "diet": ["Consume cooling foods like cucumbers and melons.", "Avoid fried and spicy foods."],
            "lifestyle": ["Wash face with a gentle cleanser twice daily.", "Avoid touching your face frequently."],
            "herbal_remedies": ["Neem paste for acne treatment.", "Turmeric for reducing inflammation."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "dry",
        "dark_circles": "absent",
        "spots": [{"location": "x=40, y=200", "size": 4}]
    },
    "nail_analysis": {
        "color": {"RGB": [160, 140, 120], "Hex": "#a08c78"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Square"]}
    },
    "outputs": {
        "disease": "eczema",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "mildly imbalanced"
        },
        "observations": [
            "Dry skin texture and spots suggest severe Vata imbalance.",
            "Square nail shape reflects mild Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Consume warm soups and stews.", "Avoid cold and raw foods."],
            "lifestyle": ["Apply moisturizing creams to affected areas.", "Avoid exposure to harsh soaps or detergents."],
            "herbal_remedies": ["Aloe vera gel for hydration.", "Ashwagandha to reduce dryness."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "rough",
        "dark_circles": "present",
        "spots": [{"location": "x=25, y=250", "size": 7}]
    },
    "nail_analysis": {
        "color": {"RGB": [80, 70, 60], "Hex": "#50463c"},
        "texture": {"edge_intensity": 0.5},
        "shape": {"shapes_detected": ["Almond"]}
    },
    "outputs": {
        "disease": "lichen planus",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "balanced"
        },
        "observations": [
            "Rough skin and dark spots indicate severe Vata imbalance.",
            "Almond nail shape reflects balanced Kapha and Pitta."
        ],
        "recommendations": {
            "diet": ["Consume warm, nourishing foods like stews.", "Avoid caffeine and alcohol."],
            "lifestyle": ["Apply natural oils like coconut oil to hydrate skin.", "Engage in stress-relieving activities like yoga."],
            "herbal_remedies": ["Neem oil for skin irritation.", "Chamomile tea to reduce inflammation."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "smooth",
        "dark_circles": "absent",
        "spots": [{"location": "x=30, y=150", "size": 3}]
    },
    "nail_analysis": {
        "color": {"RGB": [220, 200, 180], "Hex": "#dcc8b4"},
        "texture": {"edge_intensity": 0.2},
        "shape": {"shapes_detected": ["Oval"]}
    },
    "outputs": {
        "disease": "rosacea",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "balanced"
        },
        "observations": [
            "Smooth skin with redness suggests mild Pitta imbalance.",
            "Oval nail shape reflects balanced Kapha and Vata doshas."
        ],
        "recommendations": {
            "diet": ["Consume hydrating foods like cucumber and yogurt.", "Avoid hot and spicy foods."],
            "lifestyle": ["Use gentle cleansers to avoid skin irritation.", "Protect skin from extreme heat or cold."],
            "herbal_remedies": ["Chamomile tea to soothe redness.", "Aloe vera gel for hydration."]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "dry",
        "dark_circles": "present",
        "spots": [{"location": "x=40, y=250", "size": 8}]
    },
    "nail_analysis": {
        "color": {"RGB": [80, 50, 40], "Hex": "#503228"},
        "texture": {"edge_intensity": 0.2},
        "shape": {"shapes_detected": ["Square"]}
    },
    "outputs": {
        "disease": "eczema",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Dry, flaky skin indicates severe Vata imbalance.",
            "Square nails reflect a slight Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Consume warm, nourishing foods like soups and ghee.", "Avoid cold, dry, and raw foods."],
            "lifestyle": ["Use sesame oil for skin hydration before bathing.", "Avoid long hot showers that dry the skin further."],
            "herbal_remedies": [
                "Kumari (Aloe vera) gel for moisturizing the skin.",
                "Manjistha for blood purification.",
                "Licorice root powder mixed with honey for internal cooling."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "smooth",
        "dark_circles": "absent",
        "spots": [{"location": "x=25, y=150", "size": 2}]
    },
    "nail_analysis": {
        "color": {"RGB": [200, 180, 150], "Hex": "#c8b496"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Round"]}
    },
    "outputs": {
        "disease": "vitiligo",
        "dosha_analysis": {
            "vata": "mildly imbalanced",
            "pitta": "balanced",
            "kapha": "balanced"
        },
        "observations": [
            "Depigmented skin patches suggest mild Vata imbalance.",
            "Round nails reflect overall dosha balance."
        ],
        "recommendations": {
            "diet": ["Include antioxidant-rich foods like berries and spinach.", "Avoid sour foods like pickles and citrus fruits."],
            "lifestyle": ["Apply sunscreen to depigmented skin before sunlight exposure.", "Practice breathing exercises to reduce stress."],
            "herbal_remedies": [
                "Bakuchi oil for topical application on affected areas.",
                "Neem capsules for detoxification.",
                "Giloy (Tinospora cordifolia) for improving immunity."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "oily",
        "dark_circles": "present",
        "spots": [{"location": "x=50, y=200", "size": 5}]
    },
    "nail_analysis": {
        "color": {"RGB": [170, 140, 120], "Hex": "#aa8c78"},
        "texture": {"edge_intensity": 0.6},
        "shape": {"shapes_detected": ["Oval"]}
    },
    "outputs": {
        "disease": "acne",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Oily skin texture and multiple spots suggest mild Pitta imbalance.",
            "Oval nails reflect slight Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Include cooling foods like cucumber, coconut water, and bitter gourd.", "Avoid spicy, fried, and oily foods."],
            "lifestyle": ["Wash the face twice daily with a neem-based cleanser.", "Avoid touching your face frequently."],
            "herbal_remedies": [
                "Turmeric and sandalwood paste for acne.",
                "Sariva (Indian sarsaparilla) for cooling the body.",
                "Triphala powder for detoxifying the body."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "rough",
        "dark_circles": "absent",
        "spots": [{"location": "x=30, y=300", "size": 9}]
    },
    "nail_analysis": {
        "color": {"RGB": [110, 80, 70], "Hex": "#6e5046"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Almond"]}
    },
    "outputs": {
        "disease": "keratosis pilaris",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "balanced"
        },
        "observations": [
            "Rough skin texture with small bumps suggests severe Vata imbalance.",
            "Almond nails reflect balanced Pitta and Kapha conditions."
        ],
        "recommendations": {
            "diet": ["Consume warm, cooked foods to balance Vata.", "Avoid cold and dry foods."],
            "lifestyle": ["Gently exfoliate the skin with a natural scrub.", "Apply coconut oil daily to moisturize the skin."],
            "herbal_remedies": [
                "Aloe vera gel for hydration.",
                "Ashwagandha powder to calm the skin.",
                "Shatavari to promote healthy skin internally."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "normal",
        "dark_circles": "present",
        "spots": [{"location": "x=20, y=150", "size": 6}]
    },
    "nail_analysis": {
        "color": {"RGB": [150, 120, 100], "Hex": "#967864"},
        "texture": {"edge_intensity": 0.4},
        "shape": {"shapes_detected": ["Square"]}
    },
    "outputs": {
        "disease": "hyperpigmentation",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "severely imbalanced",
            "kapha": "balanced"
        },
        "observations": [
            "Smooth skin with dark pigmentation suggests severe Pitta imbalance.",
            "Square nails indicate Kapha balance."
        ],
        "recommendations": {
            "diet": ["Consume antioxidant-rich foods like pomegranate, carrots, and turmeric.", "Avoid caffeine and alcohol."],
            "lifestyle": ["Use sunscreen with herbal ingredients like turmeric daily.", "Avoid excessive sun exposure."],
            "herbal_remedies": [
                "Chandan (sandalwood) paste for topical application.",
                "Sariva (Indian sarsaparilla) for cooling the body internally.",
                "Amla juice to lighten pigmentation naturally."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "oily",
        "dark_circles": "present",
        "spots": [{"location": "x=40, y=250", "size": 4}]
    },
    "nail_analysis": {
        "color": {"RGB": [210, 190, 170], "Hex": "#d2bea8"},
        "texture": {"edge_intensity": 0.5},
        "shape": {"shapes_detected": ["Oval"]}
    },
    "outputs": {
        "disease": "melasma",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "severely imbalanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Oily skin with pigmentation suggests severe Pitta imbalance and slight Kapha imbalance.",
            "Oval nails reflect slight Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Consume antioxidant-rich foods like green tea, amla, and spinach.", "Avoid spicy foods and fried foods."],
            "lifestyle": ["Use sunscreen daily to prevent further pigmentation.", "Avoid heat exposure and intense sunlight."],
            "herbal_remedies": [
                "Licorice root for lightening pigmentation.",
                "Aloe vera to soothe the skin.",
                "Turmeric and milk paste for spot treatment."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "dry",
        "dark_circles": "present",
        "spots": [{"location": "x=30, y=150", "size": 5}]
    },
    "nail_analysis": {
        "color": {"RGB": [210, 180, 150], "Hex": "#d2b496"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Round"]}
    },
    "outputs": {
        "disease": "Eczema (Vicharchika)",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Dry, flaky skin indicates severe Vata imbalance.",
            "Round nail shape indicates slight Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Consume warm, moist foods like soups and stews.", "Avoid cold and dry foods."],
            "lifestyle": ["Apply Neem oil to affected areas before sleeping.", "Avoid harsh soaps and detergents."],
            "herbal_remedies": [
                "Triphala powder to detoxify the body.",
                "Aloe vera gel to soothe inflammation.",
                "Turmeric milk for anti-inflammatory effects."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "oily",
        "dark_circles": "absent",
        "spots": [{"location": "x=40, y=220", "size": 4}]
    },
    "nail_analysis": {
        "color": {"RGB": [180, 160, 130], "Hex": "#b49c82"},
        "texture": {"edge_intensity": 0.5},
        "shape": {"shapes_detected": ["Oval"]}
    },
    "outputs": {
        "disease": "Acne (Yauvan Pidika)",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Oily skin texture and pimples suggest mild Pitta imbalance.",
            "Oval nails indicate slight Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Include bitter foods like karela (bitter gourd).", "Avoid fried and spicy foods."],
            "lifestyle": ["Wash the face twice daily with herbal cleansers.", "Avoid touching or picking at acne."],
            "herbal_remedies": [
                "Sandalwood paste with rose water for cooling.",
                "Neem capsules to purify the blood.",
                "Apply a paste of turmeric to acne-prone areas."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "rough",
        "dark_circles": "absent",
        "spots": [{"location": "x=50, y=250", "size": 7}]
    },
    "nail_analysis": {
        "color": {"RGB": [90, 70, 50], "Hex": "#5a4632"},
        "texture": {"edge_intensity": 0.4},
        "shape": {"shapes_detected": ["Square"]}
    },
    "outputs": {
        "disease": "Psoriasis (Kitibha)",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "mildly imbalanced",
            "kapha": "balanced"
        },
        "observations": [
            "Thick, scaly skin patches indicate severe Vata imbalance.",
            "Square nails suggest balanced Kapha condition."
        ],
        "recommendations": {
            "diet": ["Avoid dairy and cold foods.", "Include anti-inflammatory foods like turmeric milk."],
            "lifestyle": ["Apply coconut oil mixed with turmeric to affected areas.", "Practice meditation to reduce stress."],
            "herbal_remedies": [
                "Guggulu for reducing inflammation.",
                "Neem oil for topical application.",
                "Triphala powder to detoxify the digestive system."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "normal",
        "dark_circles": "present",
        "spots": [{"location": "x=20, y=150", "size": 3}]
    },
    "nail_analysis": {
        "color": {"RGB": [210, 190, 170], "Hex": "#d2bea8"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Round"]}
    },
    "outputs": {
        "disease": "Vitiligo (Shwitra)",
        "dosha_analysis": {
            "vata": "mildly imbalanced",
            "pitta": "balanced",
            "kapha": "balanced"
        },
        "observations": [
            "Depigmented skin patches indicate mild Vata imbalance.",
            "Round nails suggest no major dosha imbalance."
        ],
        "recommendations": {
            "diet": ["Consume iron-rich foods like spinach.", "Avoid sour foods like pickles and citrus fruits."],
            "lifestyle": ["Apply Bakuchi oil on white patches.", "Avoid prolonged sun exposure."],
            "herbal_remedies": [
                "Turmeric and mustard oil paste for re-pigmentation.",
                "Neem capsules for detoxification.",
                "Ashwagandha for stress management."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "oily",
        "dark_circles": "absent",
        "spots": [{"location": "x=60, y=300", "size": 6}]
    },
    "nail_analysis": {
        "color": {"RGB": [100, 80, 70], "Hex": "#64503c"},
        "texture": {"edge_intensity": 0.4},
        "shape": {"shapes_detected": ["Square"]}
    },
    "outputs": {
        "disease": "Dandruff (Darunaka)",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Flaky scalp and oily texture indicate mild Pitta imbalance.",
            "Square nails suggest slightly imbalanced Kapha."
        ],
        "recommendations": {
            "diet": ["Consume foods rich in vitamin C like amla.", "Avoid dairy and fried foods."],
            "lifestyle": ["Use neem water rinse for the scalp.", "Apply warm coconut oil mixed with lemon juice."],
            "herbal_remedies": [
                "Shikakai as a natural shampoo.",
                "Amla powder for internal cleansing.",
                "Bhringraj oil for scalp massage."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "smooth",
        "dark_circles": "present",
        "spots": [{"location": "x=45, y=200", "size": 4}]
    },
    "nail_analysis": {
        "color": {"RGB": [150, 130, 110], "Hex": "#967c6e"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Oval"]}
    },
    "outputs": {
        "disease": "Hair Fall (Khalitya)",
        "dosha_analysis": {
            "vata": "mildly imbalanced",
            "pitta": "balanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Smooth skin indicates no major dosha imbalance affecting the skin.",
            "Oval nails suggest slight Kapha imbalance linked to hair fall."
        ],
        "recommendations": {
            "diet": ["Consume protein-rich foods like lentils and almonds.", "Avoid excessive caffeine and sugar."],
            "lifestyle": ["Massage the scalp with bhringraj oil regularly.", "Avoid using chemical-based shampoos."],
            "herbal_remedies": [
                "Brahmi powder for scalp nourishment.",
                "Triphala powder for balancing doshas.",
                "Fenugreek seeds soaked overnight for hair strengthening."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "dry",
        "dark_circles": "present",
        "spots": [{"location": "x=30, y=150", "size": 5}]
    },
    "nail_analysis": {
        "color": {"RGB": [210, 180, 150], "Hex": "#d2b496"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Round"]}
    },
    "outputs": {
        "disease": "Eczema (Vicharchika)",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Dry, itchy, and inflamed skin indicates severe Vata imbalance.",
            "Round nail shape reflects slight Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Consume warm, moist foods like soups and stews.", "Avoid cold, dry, and processed foods."],
            "lifestyle": ["Apply Neem oil to affected areas before sleeping.", "Avoid harsh soaps and detergents."],
            "herbal_remedies": [
                "Neem oil or paste for topical application.",
                "Turmeric milk for anti-inflammatory effects.",
                "Triphala powder to detoxify the body.",
                "Aloe vera gel to soothe the skin."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "oily",
        "dark_circles": "absent",
        "spots": [{"location": "x=40, y=220", "size": 4}]
    },
    "nail_analysis": {
        "color": {"RGB": [180, 160, 130], "Hex": "#b49c82"},
        "texture": {"edge_intensity": 0.5},
        "shape": {"shapes_detected": ["Oval"]}
    },
    "outputs": {
        "disease": "Acne (Yauvan Pidika)",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Oily skin texture and pimples suggest mild Pitta imbalance.",
            "Oval nails indicate slight Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Include bitter foods like karela (bitter gourd).", "Avoid fried and spicy foods."],
            "lifestyle": ["Wash the face twice daily with herbal cleansers.", "Avoid touching or picking at acne."],
            "herbal_remedies": [
                "Sandalwood paste and rose water for cooling.",
                "Neem capsules or powder for cleansing.",
                "A diet rich in bitter foods like karela."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "rough",
        "dark_circles": "absent",
        "spots": [{"location": "x=50, y=250", "size": 7}]
    },
    "nail_analysis": {
        "color": {"RGB": [90, 70, 50], "Hex": "#5a4632"},
        "texture": {"edge_intensity": 0.4},
        "shape": {"shapes_detected": ["Square"]}
    },
    "outputs": {
        "disease": "Psoriasis (Kitibha)",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "mildly imbalanced",
            "kapha": "balanced"
        },
        "observations": [
            "Thick, scaly skin patches indicate severe Vata imbalance.",
            "Square nails suggest balanced Kapha condition."
        ],
        "recommendations": {
            "diet": ["Avoid dairy and cold foods.", "Include anti-inflammatory foods like turmeric milk."],
            "lifestyle": ["Apply coconut oil mixed with turmeric to affected areas.", "Practice meditation to reduce stress."],
            "herbal_remedies": [
                "Panchakarma detoxification therapy.",
                "Application of coconut oil mixed with turmeric.",
                "Guggulu for reducing inflammation."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "normal",
        "dark_circles": "present",
        "spots": [{"location": "x=20, y=150", "size": 3}]
    },
    "nail_analysis": {
        "color": {"RGB": [210, 190, 170], "Hex": "#d2bea8"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Round"]}
    },
    "outputs": {
        "disease": "Vitiligo (Shwitra)",
        "dosha_analysis": {
            "vata": "mildly imbalanced",
            "pitta": "balanced",
            "kapha": "balanced"
        },
        "observations": [
            "Depigmented skin patches indicate mild Vata imbalance.",
            "Round nails suggest no major dosha imbalance."
        ],
        "recommendations": {
            "diet": ["Consume iron-rich foods like spinach.", "Avoid sour foods like pickles and citrus fruits."],
            "lifestyle": ["Apply Bakuchi oil on white patches.", "Avoid prolonged sun exposure."],
            "herbal_remedies": [
                "Bakuchi oil for topical use.",
                "Turmeric and mustard oil paste for re-pigmentation.",
                "Ashwagandha for stress reduction."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "oily",
        "dark_circles": "absent",
        "spots": [{"location": "x=60, y=300", "size": 6}]
    },
    "nail_analysis": {
        "color": {"RGB": [100, 80, 70], "Hex": "#64503c"},
        "texture": {"edge_intensity": 0.4},
        "shape": {"shapes_detected": ["Square"]}
    },
    "outputs": {
        "disease": "Dandruff (Darunaka)",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Flaky scalp and oily texture indicate mild Pitta imbalance.",
            "Square nails suggest slightly imbalanced Kapha."
        ],
        "recommendations": {
            "diet": ["Consume foods rich in vitamin C like amla.", "Avoid dairy and fried foods."],
            "lifestyle": ["Use neem water rinse for the scalp.", "Apply warm coconut oil mixed with lemon juice."],
            "herbal_remedies": [
                "Shikakai as a natural shampoo.",
                "Amla powder for internal cleansing.",
                "Neem water rinse for the scalp."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "smooth",
        "dark_circles": "present",
        "spots": [{"location": "x=45, y=200", "size": 4}]
    },
    "nail_analysis": {
        "color": {"RGB": [150, 130, 110], "Hex": "#967c6e"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Oval"]}
    },
    "outputs": {
        "disease": "Hair Fall (Khalitya)",
        "dosha_analysis": {
            "vata": "mildly imbalanced",
            "pitta": "balanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Smooth skin indicates no major dosha imbalance affecting the skin.",
            "Oval nails suggest slight Kapha imbalance linked to hair fall."
        ],
        "recommendations": {
            "diet": ["Consume protein-rich foods like lentils and almonds.", "Avoid excessive caffeine and sugar."],
            "lifestyle": ["Massage the scalp with bhringraj oil regularly.", "Avoid using chemical-based shampoos."],
            "herbal_remedies": [
                "Bhringraj oil massage to strengthen roots.",
                "Brahmi for nourishing the scalp.",
                "Triphala powder to balance the doshas."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "dry",
        "dark_circles": "absent",
        "spots": [{"location": "x=20, y=150", "size": 3}]
    },
    "nail_analysis": {
        "color": {"RGB": [180, 160, 140], "Hex": "#b49c8c"},
        "texture": {"edge_intensity": 0.5},
        "shape": {"shapes_detected": ["Oval"]},
        "growth_rate": "slow",
        "thickness": "thin",
        "surface_changes": ["longitudinal ridges"]
    },
    "outputs": {
        "disease": "Age-related brittle nails",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "balanced"
        },
        "observations": [
            "Slow nail growth and longitudinal ridges indicate aging-related changes.",
            "Dry skin suggests a Vata imbalance."
        ],
        "recommendations": {
            "diet": ["Consume warm, nourishing foods like ghee and soups.", "Avoid cold and dry foods."],
            "lifestyle": ["Apply sesame oil to nails and skin to prevent dryness.", "Avoid frequent wetting and drying of nails."],
            "herbal_remedies": [
                "Triphala powder to detoxify the body.",
                "Aloe vera gel for hydration.",
                "Ashwagandha to balance Vata and reduce brittleness."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "normal",
        "dark_circles": "present",
        "spots": [{"location": "x=30, y=200", "size": 2}]
    },
    "nail_analysis": {
        "color": {"RGB": [200, 180, 160], "Hex": "#c8b4a0"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Round"]},
        "growth_rate": "normal",
        "thickness": "thick",
        "surface_changes": ["yellow discoloration"]
    },
    "outputs": {
        "disease": "Yellow Nail Syndrome",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "imbalanced"
        },
        "observations": [
            "Yellow, thickened nails suggest Yellow Nail Syndrome.",
            "Dark circles indicate mild Pitta imbalance."
        ],
        "recommendations": {
            "diet": ["Consume bitter foods like karela and neem leaves.", "Avoid greasy and fried foods."],
            "lifestyle": ["Practice breathing exercises to improve oxygenation.", "Avoid exposure to damp environments."],
            "herbal_remedies": [
                "Neem capsules for detoxification.",
                "Turmeric milk for reducing inflammation.",
                "Tulsi tea to support respiratory health."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "oily",
        "dark_circles": "absent",
        "spots": [{"location": "x=40, y=250", "size": 5}]
    },
    "nail_analysis": {
        "color": {"RGB": [90, 70, 50], "Hex": "#5a4632"},
        "texture": {"edge_intensity": 0.6},
        "shape": {"shapes_detected": ["Square"]},
        "growth_rate": "slow",
        "thickness": "thick",
        "surface_changes": ["discoloration", "fungal infection"]
    },
    "outputs": {
        "disease": "Fungal nail infection",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "severely imbalanced"
        },
        "observations": [
            "Thick, discolored nails indicate fungal infection.",
            "Oily skin suggests Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Avoid sugary and processed foods.", "Include bitter foods like neem and turmeric."],
            "lifestyle": ["Keep nails dry and clean.", "Avoid wearing tight shoes."],
            "herbal_remedies": [
                "Neem oil for topical application.",
                "Triphala powder to cleanse the body.",
                "Turmeric paste for antifungal effects."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "dry",
        "dark_circles": "present",
        "spots": [{"location": "x=20, y=150", "size": 3}]
    },
    "nail_analysis": {
        "color": {"RGB": [150, 130, 110], "Hex": "#967c6e"},
        "texture": {"edge_intensity": 0.4},
        "shape": {"shapes_detected": ["Oval"]},
        "growth_rate": "slow",
        "thickness": "thin",
        "surface_changes": ["longitudinal ridges"]
    },
    "outputs": {
        "disease": "Age-related nail thinning",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "balanced"
        },
        "observations": [
            "Thin nails with longitudinal ridges indicate aging-related changes.",
            "Dry skin suggests a Vata imbalance."
        ],
        "recommendations": {
            "diet": ["Consume warm, nourishing foods like ghee and soups.", "Avoid cold and dry foods."],
            "lifestyle": ["Apply sesame oil to nails and skin to prevent dryness.", "Avoid frequent wetting and drying of nails."],
            "herbal_remedies": [
                "Triphala powder to detoxify the body.",
                "Aloe vera gel for hydration.",
                "Ashwagandha to balance Vata and reduce brittleness."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "normal",
        "dark_circles": "absent",
        "spots": [{"location": "x=30, y=200", "size": 2}]
    },
    "nail_analysis": {
        "color": {"RGB": [200, 180, 160], "Hex": "#c8b4a0"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Round"]},
        "growth_rate": "normal",
        "thickness": "thick",
        "surface_changes": ["yellow discoloration"]
    },
    "outputs": {
        "disease": "Yellow Nail Syndrome",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "imbalanced"
        },
        "observations": [
            "Yellow, thickened nails suggest Yellow Nail Syndrome.",
            "Dark circles indicate mild Pitta imbalance."
        ],
        "recommendations": {
            "diet": ["Consume bitter foods like karela and neem leaves.", "Avoid greasy and fried foods."],
            "lifestyle": ["Practice breathing exercises to improve oxygenation.", "Avoid exposure to damp environments."],
            "herbal_remedies": [
                "Neem capsules for detoxification.",
                "Turmeric milk for reducing inflammation.",
                "Tulsi tea to support respiratory health."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "normal",
        "dark_circles": "absent",
        "spots": [{"location": "x=20, y=180", "size": 4}]
    },
    "nail_analysis": {
        "color": {"RGB": [170, 150, 130], "Hex": "#aa9682"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Square"]},
        "growth_rate": "normal",
        "thickness": "thick",
        "surface_changes": ["horizontal ridges"]
    },
    "outputs": {
        "disease": "Beau's Lines (Horizontal Nail Ridges)",
        "dosha_analysis": {
            "vata": "mildly imbalanced",
            "pitta": "balanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Horizontal ridges suggest temporary disruption in nail growth due to stress or illness.",
            "Normal skin texture indicates no severe dosha imbalance."
        ],
        "recommendations": {
            "diet": ["Consume foods high in biotin, like eggs, nuts, and spinach.", "Drink plenty of water to stay hydrated."],
            "lifestyle": ["Reduce stress through yoga or meditation.", "Avoid harsh nail treatments or manicures."],
            "herbal_remedies": [
                "Ashwagandha to manage stress.",
                "Triphala powder for detoxification.",
                "Brahmi for improving overall health and reducing imbalances."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "oily",
        "dark_circles": "present",
        "spots": [{"location": "x=30, y=220", "size": 5}]
    },
    "nail_analysis": {
        "color": {"RGB": [90, 70, 60], "Hex": "#5a463c"},
        "texture": {"edge_intensity": 0.6},
        "shape": {"shapes_detected": ["Oval"]},
        "growth_rate": "slow",
        "thickness": "thick",
        "surface_changes": ["yellow discoloration", "fungal infection"]
    },
    "outputs": {
        "disease": "Fungal Nail Infection",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "severely imbalanced"
        },
        "observations": [
            "Thick, discolored nails indicate fungal infection.",
            "Oily skin and dark circles suggest mild Pitta imbalance."
        ],
        "recommendations": {
            "diet": ["Avoid sugary and processed foods.", "Include bitter foods like neem and turmeric."],
            "lifestyle": ["Keep nails clean and dry.", "Avoid wearing tight shoes."],
            "herbal_remedies": [
                "Apply neem oil to affected nails.",
                "Use turmeric paste to reduce inflammation.",
                "Take Triphala powder for internal detoxification."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "dry",
        "dark_circles": "present",
        "spots": [{"location": "x=40, y=160", "size": 3}]
    },
    "nail_analysis": {
        "color": {"RGB": [210, 190, 170], "Hex": "#d2bea8"},
        "texture": {"edge_intensity": 0.2},
        "shape": {"shapes_detected": ["Round"]},
        "growth_rate": "slow",
        "thickness": "thin",
        "surface_changes": ["longitudinal ridges"]
    },
    "outputs": {
        "disease": "Age-related Nail Changes",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "balanced"
        },
        "observations": [
            "Thin nails with longitudinal ridges indicate aging-related changes.",
            "Dry skin suggests severe Vata imbalance."
        ],
        "recommendations": {
            "diet": ["Consume warm, nourishing foods like ghee, soups, and whole grains.", "Avoid cold and dry foods."],
            "lifestyle": ["Apply sesame oil to nails and skin for hydration.", "Avoid repeated wetting and drying of nails."],
            "herbal_remedies": [
                "Aloe vera gel for hydration.",
                "Ashwagandha to reduce brittleness and balance Vata.",
                "Triphala powder for internal cleansing."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "normal",
        "dark_circles": "absent",
        "spots": [{"location": "x=50, y=200", "size": 4}]
    },
    "nail_analysis": {
        "color": {"RGB": [150, 130, 110], "Hex": "#967c6e"},
        "texture": {"edge_intensity": 0.4},
        "shape": {"shapes_detected": ["Square"]},
        "growth_rate": "normal",
        "thickness": "thick",
        "surface_changes": ["white spots"]
    },
    "outputs": {
        "disease": "Leukonychia (White Spots on Nails)",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "balanced"
        },
        "observations": [
            "White spots suggest temporary trauma to the nail or nutritional deficiency.",
            "Normal skin texture indicates no severe dosha imbalance."
        ],
        "recommendations": {
            "diet": ["Include zinc-rich foods like pumpkin seeds and lentils.", "Consume biotin-rich foods like eggs and walnuts."],
            "lifestyle": ["Avoid biting nails or exposing them to harsh chemicals.", "Massage nails with coconut oil to strengthen them."],
            "herbal_remedies": [
                "Neem capsules to cleanse the body.",
                "Brahmi powder to balance Pitta and improve nail health.",
                "Triphala powder for detoxification."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "oily",
        "dark_circles": "present",
        "spots": [{"location": "x=25, y=250", "size": 6}]
    },
    "nail_analysis": {
        "color": {"RGB": [100, 80, 70], "Hex": "#64503c"},
        "texture": {"edge_intensity": 0.5},
        "shape": {"shapes_detected": ["Oval"]},
        "growth_rate": "normal",
        "thickness": "thick",
        "surface_changes": ["yellow coloration"]
    },
    "outputs": {
        "disease": "Yellow Nail Syndrome",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "imbalanced"
        },
        "observations": [
            "Yellow, thickened nails suggest Yellow Nail Syndrome.",
            "Oily skin reflects Kapha imbalance."
        ],
        "recommendations": {
            "diet": ["Include bitter foods like neem leaves and karela.", "Avoid greasy and fried foods."],
            "lifestyle": ["Practice breathing exercises for better oxygenation.", "Keep nails clean and avoid harsh chemicals."],
            "herbal_remedies": [
                "Neem oil for topical application.",
                "Turmeric milk for reducing inflammation.",
                "Tulsi tea to support respiratory health."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "dry",
        "dark_circles": "present",
        "spots": [{"location": "x=30, y=150", "size": 4}]
    },
    "nail_analysis": {
        "color": {"RGB": [210, 180, 150], "Hex": "#d2b496"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Round"]},
        "growth_rate": "slow",
        "thickness": "thin",
        "surface_changes": ["longitudinal ridges"]
    },
    "outputs": {
        "disease": "Eczema (Vicharchika)",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Dry, itchy, and inflamed skin indicates severe Vata imbalance.",
            "Longitudinal ridges on nails suggest aging-related changes."
        ],
        "recommendations": {
            "diet": ["Consume warm, moist foods like soups and stews.", "Avoid cold, dry, and processed foods."],
            "lifestyle": ["Apply Khadira (Acacia catechu) paste to affected areas.", "Avoid harsh soaps and detergents."],
            "herbal_remedies": [
                "Neem oil for topical application.",
                "Triphala powder to detoxify the body.",
                "Aloe vera gel to soothe the skin.",
                "Shatavari to balance Vata and improve hydration."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "oily",
        "dark_circles": "absent",
        "spots": [{"location": "x=40, y=220", "size": 5}]
    },
    "nail_analysis": {
        "color": {"RGB": [180, 160, 130], "Hex": "#b49c82"},
        "texture": {"edge_intensity": 0.5},
        "shape": {"shapes_detected": ["Oval"]},
        "growth_rate": "normal",
        "thickness": "thick",
        "surface_changes": ["yellow discoloration"]
    },
    "outputs": {
        "disease": "Acne (Yauvan Pidika)",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Oily skin texture and pimples suggest mild Pitta imbalance.",
            "Yellow discoloration on nails indicates possible fungal infection."
        ],
        "recommendations": {
            "diet": ["Include bitter foods like karela (bitter melon) and neem leaves.", "Avoid fried and spicy foods."],
            "lifestyle": ["Wash the face twice daily with herbal cleansers.", "Avoid touching or picking at acne."],
            "herbal_remedies": [
                "Sandalwood paste and rose water for cooling.",
                "Neem capsules for cleansing the blood.",
                "Pippali (long pepper) powder to reduce inflammation."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "rough",
        "dark_circles": "absent",
        "spots": [{"location": "x=50, y=250", "size": 7}]
    },
    "nail_analysis": {
        "color": {"RGB": [90, 70, 50], "Hex": "#5a4632"},
        "texture": {"edge_intensity": 0.4},
        "shape": {"shapes_detected": ["Square"]},
        "growth_rate": "slow",
        "thickness": "thick",
        "surface_changes": ["white spots"]
    },
    "outputs": {
        "disease": "Psoriasis (Kitibha)",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "mildly imbalanced",
            "kapha": "balanced"
        },
        "observations": [
            "Thick, scaly skin patches indicate severe Vata imbalance.",
            "White spots on nails suggest nutritional deficiencies."
        ],
        "recommendations": {
            "diet": ["Avoid dairy and cold foods.", "Include anti-inflammatory foods like turmeric milk."],
            "lifestyle": ["Apply coconut oil mixed with turmeric to affected areas.", "Practice meditation to reduce stress."],
            "herbal_remedies": [
                "Guggulu to reduce inflammation.",
                "Khadira (Acacia catechu) for skin purification.",
                "Moringa powder to improve nutrition and immunity."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "normal",
        "dark_circles": "present",
        "spots": [{"location": "x=20, y=150", "size": 3}]
    },
    "nail_analysis": {
        "color": {"RGB": [210, 190, 170], "Hex": "#d2bea8"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Round"]},
        "growth_rate": "normal",
        "thickness": "thin",
        "surface_changes": ["longitudinal ridges"]
    },
    "outputs": {
        "disease": "Vitiligo (Shwitra)",
        "dosha_analysis": {
            "vata": "mildly imbalanced",
            "pitta": "balanced",
            "kapha": "balanced"
        },
        "observations": [
            "Depigmented skin patches indicate mild Vata imbalance.",
            "Longitudinal ridges on nails suggest aging-related changes."
        ],
        "recommendations": {
            "diet": ["Consume iron-rich foods like spinach and lentils.", "Avoid sour foods like pickles and citrus fruits."],
            "lifestyle": ["Apply Bakuchi oil on white patches.", "Avoid prolonged sun exposure."],
            "herbal_remedies": [
                "Turmeric and mustard oil paste for re-pigmentation.",
                "Ashwagandha for stress reduction.",
                "Fenugreek seeds soaked overnight for improving skin health."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "oily",
        "dark_circles": "present",
        "spots": [{"location": "x=35, y=200", "size": 6}]
    },
    "nail_analysis": {
        "color": {"RGB": [150, 130, 110], "Hex": "#967c6e"},
        "texture": {"edge_intensity": 0.4},
        "shape": {"shapes_detected": ["Oval"]},
        "growth_rate": "slow",
        "thickness": "thick",
        "surface_changes": ["yellow discoloration"]
    },
    "outputs": {
        "disease": "Dandruff (Darunaka)",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "imbalanced"
        },
        "observations": [
            "Flaky scalp and oily texture indicate mild Pitta imbalance.",
            "Yellow discoloration on nails suggests fungal infection."
        ],
        "recommendations": {
            "diet": ["Consume foods rich in vitamin C like amla.", "Avoid dairy and fried foods."],
            "lifestyle": ["Use neem water rinse for the scalp.", "Apply warm coconut oil mixed with lemon juice."],
            "herbal_remedies": [
                "Shikakai as a natural shampoo.",
                "Amla powder for internal cleansing.",
                "Coriander seeds for cooling the body."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "rough",
        "dark_circles": "present",
        "spots": [{"location": "x=20, y=200", "size": 5}]
    },
    "nail_analysis": {
        "color": {"RGB": [100, 80, 60], "Hex": "#64503c"},
        "texture": {"edge_intensity": 0.6},
        "shape": {"shapes_detected": ["Square"]},
        "growth_rate": "slow",
        "thickness": "thick",
        "surface_changes": ["yellow discoloration"]
    },
    "outputs": {
        "disease": "Psoriasis (Kitibha)",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "mildly imbalanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Thick, scaly, and itchy skin patches indicate severe Vata imbalance.",
            "Yellow nail discoloration suggests fungal involvement."
        ],
        "recommendations": {
            "diet": ["Avoid dairy and cold foods.", "Consume anti-inflammatory foods like turmeric milk."],
            "lifestyle": ["Apply a paste of Khadira and neem on affected areas.", "Practice meditation to reduce stress."],
            "herbal_remedies": [
                "Khadira (Acacia catechu) for skin purification.",
                "Guggulu to reduce inflammation.",
                "Moringa powder to boost immunity and nutrition."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "oily",
        "dark_circles": "absent",
        "spots": [{"location": "x=30, y=220", "size": 4}]
    },
    "nail_analysis": {
        "color": {"RGB": [200, 170, 140], "Hex": "#c8aa8c"},
        "texture": {"edge_intensity": 0.5},
        "shape": {"shapes_detected": ["Oval"]},
        "growth_rate": "normal",
        "thickness": "thin",
        "surface_changes": ["white spots"]
    },
    "outputs": {
        "disease": "Acne (Yauvan Pidika)",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Oily skin texture and pimples suggest mild Pitta imbalance.",
            "White spots on nails suggest zinc deficiency."
        ],
        "recommendations": {
            "diet": ["Include bitter foods like karela (bitter melon).", "Avoid fried and spicy foods."],
            "lifestyle": ["Wash the face twice daily with a neem-based cleanser.", "Avoid touching or picking at acne."],
            "herbal_remedies": [
                "Sandalwood paste and rose water for cooling effects.",
                "Neem capsules for cleansing the blood.",
                "Pippali (long pepper) powder to reduce inflammation."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "dry",
        "dark_circles": "present",
        "spots": [{"location": "x=40, y=180", "size": 6}]
    },
    "nail_analysis": {
        "color": {"RGB": [150, 120, 100], "Hex": "#967864"},
        "texture": {"edge_intensity": 0.4},
        "shape": {"shapes_detected": ["Round"]},
        "growth_rate": "slow",
        "thickness": "thin",
        "surface_changes": ["longitudinal ridges"]
    },
    "outputs": {
        "disease": "Eczema (Vicharchika)",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Dry, itchy, and inflamed skin indicates severe Vata imbalance.",
            "Longitudinal ridges on nails suggest aging-related changes."
        ],
        "recommendations": {
            "diet": ["Consume warm, moist foods like soups and stews.", "Avoid cold and dry foods."],
            "lifestyle": ["Apply Khadira paste to inflamed areas.", "Avoid harsh soaps and detergents."],
            "herbal_remedies": [
                "Triphala powder to detoxify the body.",
                "Aloe vera gel to soothe the skin.",
                "Shatavari to balance Vata and improve hydration."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "normal",
        "dark_circles": "absent",
        "spots": [{"location": "x=25, y=250", "size": 3}]
    },
    "nail_analysis": {
        "color": {"RGB": [90, 70, 60], "Hex": "#5a463c"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Square"]},
        "growth_rate": "normal",
        "thickness": "thick",
        "surface_changes": ["yellow discoloration"]
    },
    "outputs": {
        "disease": "Yellow Nail Syndrome",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "imbalanced"
        },
        "observations": [
            "Yellow, thickened nails suggest fungal infection or systemic imbalance.",
            "Normal skin texture indicates no severe skin involvement."
        ],
        "recommendations": {
            "diet": ["Consume bitter foods like neem and karela.", "Avoid greasy and fried foods."],
            "lifestyle": ["Practice breathing exercises for better oxygenation.", "Keep nails clean and dry."],
            "herbal_remedies": [
                "Neem oil for topical application.",
                "Turmeric milk for reducing inflammation.",
                "Coriander seeds for internal cooling."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "oily",
        "dark_circles": "present",
        "spots": [{"location": "x=45, y=190", "size": 5}]
    },
    "nail_analysis": {
        "color": {"RGB": [170, 140, 120], "Hex": "#aa8c78"},
        "texture": {"edge_intensity": 0.5},
        "shape": {"shapes_detected": ["Oval"]},
        "growth_rate": "slow",
        "thickness": "thick",
        "surface_changes": ["peeling"]
    },
    "outputs": {
        "disease": "Dandruff (Darunaka)",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "imbalanced"
        },
        "observations": [
            "Flaky scalp and oily texture indicate mild Pitta imbalance.",
            "Peeling nails suggest nutritional deficiencies."
        ],
        "recommendations": {
            "diet": ["Consume foods rich in vitamin C like amla.", "Avoid dairy and fried foods."],
            "lifestyle": ["Use neem water rinse for the scalp.", "Apply warm coconut oil mixed with lemon juice."],
            "herbal_remedies": [
                "Amla powder for internal cleansing.",
                "Shikakai as a natural shampoo.",
                "Fenugreek seeds soaked overnight for scalp health."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "dry",
        "dark_circles": "present",
        "spots": [{"location": "x=30, y=150", "size": 4}]
    },
    "nail_analysis": {
        "color": {"RGB": [180, 150, 120], "Hex": "#b49678"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Round"]},
        "growth_rate": "slow",
        "thickness": "thin",
        "surface_changes": ["longitudinal ridges"]
    },
    "outputs": {
        "disease": "Eczema (Vicharchika)",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "balanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Dry, itchy, and inflamed skin indicates severe Vata imbalance.",
            "Longitudinal ridges on nails suggest aging-related changes."
        ],
        "recommendations": {
            "diet": ["Consume warm, moist foods like soups and stews.", "Avoid cold, dry, and processed foods."],
            "lifestyle": ["Apply Khadira paste to inflamed areas.", "Avoid harsh soaps and detergents."],
            "herbal_remedies": [
                "Neem oil for topical application.",
                "Triphala powder to detoxify the body.",
                "Shatavari to balance Vata and improve hydration.",
                "Coriander seeds for cooling the body."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "oily",
        "dark_circles": "absent",
        "spots": [{"location": "x=40, y=220", "size": 5}]
    },
    "nail_analysis": {
        "color": {"RGB": [90, 70, 50], "Hex": "#5a4632"},
        "texture": {"edge_intensity": 0.5},
        "shape": {"shapes_detected": ["Oval"]},
        "growth_rate": "normal",
        "thickness": "thick",
        "surface_changes": ["yellow discoloration"]
    },
    "outputs": {
        "disease": "Acne (Yauvan Pidika)",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "slightly imbalanced"
        },
        "observations": [
            "Oily skin texture and pimples suggest mild Pitta imbalance.",
            "Yellow discoloration on nails indicates fungal involvement."
        ],
        "recommendations": {
            "diet": ["Include bitter foods like karela (bitter melon).", "Avoid fried and spicy foods."],
            "lifestyle": ["Wash the face twice daily with a neem-based cleanser.", "Avoid touching or picking at acne."],
            "herbal_remedies": [
                "Sandalwood paste and rose water for cooling.",
                "Neem capsules for cleansing the blood.",
                "Pippali (long pepper) powder to reduce inflammation.",
                "Cinnamon for improving circulation."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "light",
        "texture": "normal",
        "dark_circles": "present",
        "spots": [{"location": "x=20, y=150", "size": 3}]
    },
    "nail_analysis": {
        "color": {"RGB": [210, 190, 170], "Hex": "#d2bea8"},
        "texture": {"edge_intensity": 0.3},
        "shape": {"shapes_detected": ["Round"]},
        "growth_rate": "normal",
        "thickness": "thin",
        "surface_changes": ["white spots"]
    },
    "outputs": {
        "disease": "Vitiligo (Shwitra)",
        "dosha_analysis": {
            "vata": "mildly imbalanced",
            "pitta": "balanced",
            "kapha": "balanced"
        },
        "observations": [
            "Depigmented skin patches indicate mild Vata imbalance.",
            "White spots on nails suggest nutritional deficiencies."
        ],
        "recommendations": {
            "diet": ["Consume iron-rich foods like spinach and lentils.", "Avoid sour foods like pickles and citrus fruits."],
            "lifestyle": ["Apply Bakuchi oil on white patches.", "Avoid prolonged sun exposure."],
            "herbal_remedies": [
                "Turmeric and mustard oil paste for re-pigmentation.",
                "Ashwagandha for stress reduction.",
                "Fenugreek seeds soaked overnight for improving skin health.",
                "Saffron for enhancing skin tone."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "medium",
        "texture": "oily",
        "dark_circles": "present",
        "spots": [{"location": "x=35, y=200", "size": 6}]
    },
    "nail_analysis": {
        "color": {"RGB": [150, 130, 110], "Hex": "#967c6e"},
        "texture": {"edge_intensity": 0.4},
        "shape": {"shapes_detected": ["Oval"]},
        "growth_rate": "slow",
        "thickness": "thick",
        "surface_changes": ["peeling"]
    },
    "outputs": {
        "disease": "Dandruff (Darunaka)",
        "dosha_analysis": {
            "vata": "balanced",
            "pitta": "mildly imbalanced",
            "kapha": "imbalanced"
        },
        "observations": [
            "Flaky scalp and oily texture indicate mild Pitta imbalance.",
            "Peeling nails suggest nutritional deficiencies."
        ],
        "recommendations": {
            "diet": ["Consume foods rich in vitamin C like amla.", "Avoid dairy and fried foods."],
            "lifestyle": ["Use neem water rinse for the scalp.", "Apply warm coconut oil mixed with lemon juice."],
            "herbal_remedies": [
                "Amla powder for internal cleansing.",
                "Shikakai as a natural shampoo.",
                "Fenugreek seeds soaked overnight for scalp health.",
                "Curry leaves for improving hair strength."
            ]
        }
    }
},
{
    "skin_analysis": {
        "tone": "dark",
        "texture": "rough",
        "dark_circles": "absent",
        "spots": [{"location": "x=50, y=250", "size": 7}]
    },
    "nail_analysis": {
        "color": {"RGB": [90, 70, 50], "Hex": "#5a4632"},
        "texture": {"edge_intensity": 0.4},
        "shape": {"shapes_detected": ["Square"]},
        "growth_rate": "slow",
        "thickness": "thick",
        "surface_changes": ["yellow discoloration"]
    },
    "outputs": {
        "disease": "Psoriasis (Kitibha)",
        "dosha_analysis": {
            "vata": "severely imbalanced",
            "pitta": "mildly imbalanced",
            "kapha": "balanced"
        },
        "observations": [
            "Thick, scaly, and itchy skin patches indicate severe Vata imbalance.",
            "Yellow nail discoloration suggests fungal involvement."
        ],
        "recommendations": {
            "diet": ["Avoid dairy and cold foods.", "Consume anti-inflammatory foods like turmeric milk."],
            "lifestyle": ["Apply a paste of Khadira and neem on affected areas.", "Practice meditation to reduce stress."],
            "herbal_remedies": [
                "Khadira (Acacia catechu) for skin purification.",
                "Guggulu to reduce inflammation.",
                "Moringa powder to boost immunity and nutrition.",
                "Black pepper for improving digestion."
            ]
        }
    }
},
{
        "skin_analysis": {
            "tone": "medium",
            "texture": "dry",
            "dark_circles": "present",
            "spots": [{"location": "x=25, y=160", "size": 3}]
        },
        "nail_analysis": {
            "color": {"RGB": [180, 150, 120], "Hex": "#b49678"},
            "texture": {"edge_intensity": 0.3},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "slow",
            "thickness": "thin",
            "surface_changes": ["longitudinal ridges"]
        },
        "outputs": {
            "disease": "Eczema (Vicharchika)",
            "dosha_analysis": {
                "vata": "severely imbalanced",
                "pitta": "balanced",
                "kapha": "slightly imbalanced"
            },
            "observations": [
                "Dry, itchy, and inflamed skin indicates severe Vata imbalance.",
                "Longitudinal ridges on nails suggest aging-related changes."
            ],
            "recommendations": {
                "diet": ["Consume warm, moist foods like soups and stews.", "Avoid cold, dry, and processed foods."],
                "lifestyle": ["Apply Khadira paste to inflamed areas.", "Avoid harsh soaps and detergents."],
                "herbal_remedies": [
                    "Neem oil for topical application.",
                    "Triphala powder to detoxify the body.",
                    "Shatavari to balance Vata and improve hydration.",
                    "Coriander seeds for cooling the body."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "dark",
            "texture": "oily",
            "dark_circles": "absent",
            "spots": [{"location": "x=35, y=220", "size": 5}]
        },
        "nail_analysis": {
            "color": {"RGB": [90, 70, 50], "Hex": "#5a4632"},
            "texture": {"edge_intensity": 0.5},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "normal",
            "thickness": "thick",
            "surface_changes": ["yellow discoloration"]
        },
        "outputs": {
            "disease": "Acne (Yauvan Pidika)",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "mildly imbalanced",
                "kapha": "slightly imbalanced"
            },
            "observations": [
                "Oily skin texture and pimples suggest mild Pitta imbalance.",
                "Yellow discoloration on nails indicates fungal involvement."
            ],
            "recommendations": {
                "diet": ["Include bitter foods like karela (bitter melon).", "Avoid fried and spicy foods."],
                "lifestyle": ["Wash the face twice daily with a neem-based cleanser.", "Avoid touching or picking at acne."],
                "herbal_remedies": [
                    "Sandalwood paste and rose water for cooling.",
                    "Neem capsules for cleansing the blood.",
                    "Pippali (long pepper) powder to reduce inflammation.",
                    "Cinnamon for improving circulation."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "light",
            "texture": "normal",
            "dark_circles": "present",
            "spots": [{"location": "x=20, y=150", "size": 3}]
        },
        "nail_analysis": {
            "color": {"RGB": [210, 190, 170], "Hex": "#d2bea8"},
            "texture": {"edge_intensity": 0.3},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "normal",
            "thickness": "thin",
            "surface_changes": ["white spots"]
        },
        "outputs": {
            "disease": "Vitiligo (Shwitra)",
            "dosha_analysis": {
                "vata": "mildly imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Depigmented skin patches indicate mild Vata imbalance.",
                "White spots on nails suggest nutritional deficiencies."
            ],
            "recommendations": {
                "diet": ["Consume iron-rich foods like spinach and lentils.", "Avoid sour foods like pickles and citrus fruits."],
                "lifestyle": ["Apply Bakuchi oil on white patches.", "Avoid prolonged sun exposure."],
                "herbal_remedies": [
                    "Turmeric and mustard oil paste for re-pigmentation.",
                    "Ashwagandha for stress reduction.",
                    "Fenugreek seeds soaked overnight for improving skin health.",
                    "Saffron for enhancing skin tone."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "medium",
            "texture": "oily",
            "dark_circles": "present",
            "spots": [{"location": "x=45, y=200", "size": 6}]
        },
        "nail_analysis": {
            "color": {"RGB": [150, 130, 110], "Hex": "#967c6e"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "slow",
            "thickness": "thick",
            "surface_changes": ["peeling"]
        },
        "outputs": {
            "disease": "Dandruff (Darunaka)",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "mildly imbalanced",
                "kapha": "imbalanced"
            },
            "observations": [
                "Flaky scalp and oily texture indicate mild Pitta imbalance.",
                "Peeling nails suggest nutritional deficiencies."
            ],
            "recommendations": {
                "diet": ["Consume foods rich in vitamin C like amla.", "Avoid dairy and fried foods."],
                "lifestyle": ["Use neem water rinse for the scalp.", "Apply warm coconut oil mixed with lemon juice."],
                "herbal_remedies": [
                    "Amla powder for internal cleansing.",
                    "Shikakai as a natural shampoo.",
                    "Fenugreek seeds soaked overnight for scalp health.",
                    "Curry leaves for improving hair strength."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "dark",
            "texture": "rough",
            "dark_circles": "absent",
            "spots": [{"location": "x=50, y=250", "size": 7}]
        },
        "nail_analysis": {
            "color": {"RGB": [90, 70, 50], "Hex": "#5a4632"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Square"]},
            "growth_rate": "slow",
            "thickness": "thick",
            "surface_changes": ["yellow discoloration"]
        },
        "outputs": {
            "disease": "Psoriasis (Kitibha)",
            "dosha_analysis": {
                "vata": "severely imbalanced",
                "pitta": "mildly imbalanced",
                "kapha": "balanced"
            },
            "observations": [
                "Thick, scaly, and itchy skin patches indicate severe Vata imbalance.",
                "Yellow nail discoloration suggests fungal involvement."
            ],
            "recommendations": {
                "diet": ["Avoid dairy and cold foods.", "Consume anti-inflammatory foods like turmeric milk."],
                "lifestyle": ["Apply a paste of Khadira and neem on affected areas.", "Practice meditation to reduce stress."],
                "herbal_remedies": [
                    "Khadira (Acacia catechu) for skin purification.",
                    "Guggulu to reduce inflammation.",
                    "Moringa powder to boost immunity and nutrition.",
                    "Black pepper for improving digestion."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "light",
            "texture": "smooth",
            "dark_circles": "present",
            "spots": [{"location": "x=25, y=175", "size": 3}]
        },
        "nail_analysis": {
            "color": {"RGB": [200, 170, 140], "Hex": "#c8aa8c"},
            "texture": {"edge_intensity": 0.2},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "normal",
            "thickness": "thin",
            "surface_changes": ["white spots"]
        },
        "outputs": {
            "disease": "Leukonychia (White Spots on Nails)",
            "dosha_analysis": {
                "vata": "slightly imbalanced",
                "pitta": "balanced",
                "kapha": "slightly imbalanced"
            },
            "observations": [
                "White spots on nails suggest trauma or zinc deficiency.",
                "Smooth skin texture indicates no severe skin issues."
            ],
            "recommendations": {
                "diet": ["Include zinc-rich foods like pumpkin seeds and lentils.", "Consume biotin-rich foods like walnuts and eggs."],
                "lifestyle": ["Avoid biting nails or exposing them to harsh chemicals.", "Massage nails with coconut oil to strengthen them."],
                "herbal_remedies": [
                    "Neem capsules for detoxification.",
                    "Fenugreek seeds soaked overnight for improving nail health.",
                    "Cardamom for improving digestion and nutrient absorption."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "medium",
            "texture": "dry",
            "dark_circles": "absent",
            "spots": [{"location": "x=30, y=200", "size": 4}]
        },
        "nail_analysis": {
            "color": {"RGB": [170, 140, 120], "Hex": "#aa8c78"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "slow",
            "thickness": "thin",
            "surface_changes": ["longitudinal ridges"]
        },
        "outputs": {
            "disease": "Age-related Dry Skin and Nails",
            "dosha_analysis": {
                "vata": "severely imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Dry skin and longitudinal ridges on nails suggest aging-related Vata imbalance.",
                "Oval nail shape indicates no major structural abnormalities."
            ],
            "recommendations": {
                "diet": ["Consume warm, nourishing foods like ghee, soups, and whole grains.", "Avoid cold and dry foods."],
                "lifestyle": ["Apply sesame oil to nails and skin for hydration.", "Avoid repeated wetting and drying of nails."],
                "herbal_remedies": [
                    "Triphala powder for internal cleansing.",
                    "Shatavari to balance Vata and improve hydration.",
                    "Coriander seeds for cooling the body."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "dark",
            "texture": "oily",
            "dark_circles": "present",
            "spots": [{"location": "x=40, y=250", "size": 5}]
        },
        "nail_analysis": {
            "color": {"RGB": [100, 80, 60], "Hex": "#64503c"},
            "texture": {"edge_intensity": 0.5},
            "shape": {"shapes_detected": ["Square"]},
            "growth_rate": "normal",
            "thickness": "thick",
            "surface_changes": ["yellow discoloration"]
        },
        "outputs": {
            "disease": "Fungal Nail Infection",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "mildly imbalanced",
                "kapha": "severely imbalanced"
            },
            "observations": [
                "Thick, discolored nails suggest fungal infection.",
                "Oily skin and dark circles indicate mild Pitta imbalance."
            ],
            "recommendations": {
                "diet": ["Avoid sugary and processed foods.", "Include bitter foods like neem and turmeric."],
                "lifestyle": ["Keep nails dry and clean.", "Avoid wearing tight-fitting shoes."],
                "herbal_remedies": [
                    "Neem oil for topical application.",
                    "Moringa powder to boost immunity.",
                    "Kalonji (black seed) oil for antifungal effects."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "light",
            "texture": "normal",
            "dark_circles": "present",
            "spots": [{"location": "x=20, y=150", "size": 2}]
        },
        "nail_analysis": {
            "color": {"RGB": [210, 190, 170], "Hex": "#d2bea8"},
            "texture": {"edge_intensity": 0.2},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "normal",
            "thickness": "thin",
            "surface_changes": ["peeling"]
        },
        "outputs": {
            "disease": "Vitamin Deficiency",
            "dosha_analysis": {
                "vata": "mildly imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Peeling nails suggest vitamin deficiency or dryness.",
                "Dark circles indicate mild Vata imbalance."
            ],
            "recommendations": {
                "diet": ["Include foods rich in vitamin E like almonds and sunflower seeds.", "Consume leafy greens for balanced nutrition."],
                "lifestyle": ["Massage nails with warm coconut oil daily.", "Avoid excessive exposure to harsh detergents."],
                "herbal_remedies": [
                    "Aloe vera gel for hydration.",
                    "Ashwagandha for strengthening nails and reducing Vata imbalance.",
                    "Fenugreek seeds for improving nutrient absorption."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "medium",
            "texture": "oily",
            "dark_circles": "absent",
            "spots": [{"location": "x=30, y=200", "size": 6}]
        },
        "nail_analysis": {
            "color": {"RGB": [170, 140, 120], "Hex": "#aa8c78"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "normal",
            "thickness": "thick",
            "surface_changes": ["dark stripes"]
        },
        "outputs": {
            "disease": "Melanonychia (Dark Nail Streaks)",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "mildly imbalanced",
                "kapha": "balanced"
            },
            "observations": [
                "Dark stripes on nails may indicate melanonychia.",
                "Oily skin indicates no major dryness-related issues."
            ],
            "recommendations": {
                "diet": ["Consume antioxidant-rich foods like berries and pomegranate.", "Avoid fried and greasy foods."],
                "lifestyle": ["Keep nails moisturized with almond oil.", "Avoid trauma to the nail beds."],
                "herbal_remedies": [
                    "Turmeric for reducing inflammation.",
                    "Guggulu for improving circulation.",
                    "Saffron for enhancing skin and nail health."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "medium",
            "texture": "dry",
            "dark_circles": "present",
            "spots": [{"location": "x=30, y=150", "size": 4}]
        },
        "nail_analysis": {
            "color": {"RGB": [180, 150, 120], "Hex": "#b49678"},
            "texture": {"edge_intensity": 0.3},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "slow",
            "thickness": "thin",
            "surface_changes": ["longitudinal ridges"]
        },
        "outputs": {
            "disease": "Eczema (Vicharchika)",
            "dosha_analysis": {
                "vata": "severely imbalanced",
                "pitta": "balanced",
                "kapha": "slightly imbalanced"
            },
            "observations": [
                "Dry, itchy, and inflamed skin indicates severe Vata imbalance.",
                "Longitudinal ridges on nails suggest aging-related changes."
            ],
            "recommendations": {
                "diet": ["Consume warm, moist foods like soups and stews.", "Avoid cold, dry, and processed foods."],
                "lifestyle": ["Apply Khadira paste to inflamed areas.", "Avoid harsh soaps and detergents."],
                "herbal_remedies": [
                    "Neem oil for topical application.",
                    "Triphala powder to detoxify the body.",
                    "Shatavari to balance Vata and improve hydration.",
                    "Coriander seeds for cooling the body."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "dark",
            "texture": "oily",
            "dark_circles": "absent",
            "spots": [{"location": "x=40, y=220", "size": 5}]
        },
        "nail_analysis": {
            "color": {"RGB": [90, 70, 50], "Hex": "#5a4632"},
            "texture": {"edge_intensity": 0.5},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "normal",
            "thickness": "thick",
            "surface_changes": ["yellow discoloration"]
        },
        "outputs": {
            "disease": "Acne (Yauvan Pidika)",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "mildly imbalanced",
                "kapha": "slightly imbalanced"
            },
            "observations": [
                "Oily skin texture and pimples suggest mild Pitta imbalance.",
                "Yellow discoloration on nails indicates fungal involvement."
            ],
            "recommendations": {
                "diet": ["Include bitter foods like karela (bitter melon).", "Avoid fried and spicy foods."],
                "lifestyle": ["Wash the face twice daily with a neem-based cleanser.", "Avoid touching or picking at acne."],
                "herbal_remedies": [
                    "Sandalwood paste and rose water for cooling.",
                    "Neem capsules for cleansing the blood.",
                    "Pippali (long pepper) powder to reduce inflammation.",
                    "Cinnamon for improving circulation."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "light",
            "texture": "normal",
            "dark_circles": "present",
            "spots": [{"location": "x=20, y=150", "size": 3}]
        },
        "nail_analysis": {
            "color": {"RGB": [210, 190, 170], "Hex": "#d2bea8"},
            "texture": {"edge_intensity": 0.3},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "normal",
            "thickness": "thin",
            "surface_changes": ["white spots"]
        },
        "outputs": {
            "disease": "Vitiligo (Shwitra)",
            "dosha_analysis": {
                "vata": "mildly imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Depigmented skin patches indicate mild Vata imbalance.",
                "White spots on nails suggest nutritional deficiencies."
            ],
            "recommendations": {
                "diet": ["Consume iron-rich foods like spinach and lentils.", "Avoid sour foods like pickles and citrus fruits."],
                "lifestyle": ["Apply Bakuchi oil on white patches.", "Avoid prolonged sun exposure."],
                "herbal_remedies": [
                    "Turmeric and mustard oil paste for re-pigmentation.",
                    "Ashwagandha for stress reduction.",
                    "Fenugreek seeds soaked overnight for improving skin health.",
                    "Saffron for enhancing skin tone."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "medium",
            "texture": "oily",
            "dark_circles": "present",
            "spots": [{"location": "x=45, y=200", "size": 6}]
        },
        "nail_analysis": {
            "color": {"RGB": [150, 130, 110], "Hex": "#967c6e"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "slow",
            "thickness": "thick",
            "surface_changes": ["peeling"]
        },
        "outputs": {
            "disease": "Dandruff (Darunaka)",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "mildly imbalanced",
                "kapha": "imbalanced"
            },
            "observations": [
                "Flaky scalp and oily texture indicate mild Pitta imbalance.",
                "Peeling nails suggest nutritional deficiencies."
            ],
            "recommendations": {
                "diet": ["Consume foods rich in vitamin C like amla.", "Avoid dairy and fried foods."],
                "lifestyle": ["Use neem water rinse for the scalp.", "Apply warm coconut oil mixed with lemon juice."],
                "herbal_remedies": [
                    "Amla powder for internal cleansing.",
                    "Shikakai as a natural shampoo.",
                    "Fenugreek seeds soaked overnight for scalp health.",
                    "Curry leaves for improving hair strength."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "dark",
            "texture": "rough",
            "dark_circles": "absent",
            "spots": [{"location": "x=50, y=250", "size": 7}]
        },
        "nail_analysis": {
            "color": {"RGB": [90, 70, 50], "Hex": "#5a4632"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Square"]},
            "growth_rate": "slow",
            "thickness": "thick",
            "surface_changes": ["yellow discoloration"]
        },
        "outputs": {
            "disease": "Psoriasis (Kitibha)",
            "dosha_analysis": {
                "vata": "severely imbalanced",
                "pitta": "mildly imbalanced",
                "kapha": "balanced"
            },
            "observations": [
                "Thick, scaly, and itchy skin patches indicate severe Vata imbalance.",
                "Yellow nail discoloration suggests fungal involvement."
            ],
            "recommendations": {
                "diet": ["Avoid dairy and cold foods.", "Consume anti-inflammatory foods like turmeric milk."],
                "lifestyle": ["Apply a paste of Khadira and neem on affected areas.", "Practice meditation to reduce stress."],
                "herbal_remedies": [
                    "Khadira (Acacia catechu) for skin purification.",
                    "Guggulu to reduce inflammation.",
                    "Moringa powder to boost immunity and nutrition.",
                    "Black pepper for improving digestion."
                ]
            }
        }
    },
     {
        "skin_analysis": {
            "tone": "medium",
            "texture": "rough",
            "dark_circles": "present",
            "spots": [{"location": "x=25, y=170", "size": 4}]
        },
        "nail_analysis": {
            "color": {"RGB": [180, 160, 130], "Hex": "#b49c82"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "slow",
            "thickness": "thin",
            "surface_changes": ["longitudinal ridges"]
        },
        "outputs": {
            "disease": "Eczema (Vicharchika)",
            "dosha_analysis": {
                "vata": "severely imbalanced",
                "pitta": "balanced",
                "kapha": "slightly imbalanced"
            },
            "observations": [
                "Rough, itchy skin with longitudinal ridges on nails indicates severe Vata imbalance.",
                "Dark circles suggest dryness and dehydration."
            ],
            "recommendations": {
                "diet": ["Consume warm, moist foods like soups and ghee.", "Avoid cold, dry, and processed foods."],
                "lifestyle": ["Apply Khadira paste or aloe vera gel to affected areas.", "Avoid harsh soaps and detergents."],
                "herbal_remedies": [
                    "Neem oil for topical application.",
                    "Triphala powder to detoxify the body.",
                    "Shatavari to balance Vata and improve hydration.",
                    "Coriander powder for cooling effects."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "dark",
            "texture": "oily",
            "dark_circles": "absent",
            "spots": [{"location": "x=40, y=190", "size": 6}]
        },
        "nail_analysis": {
            "color": {"RGB": [90, 70, 50], "Hex": "#5a4632"},
            "texture": {"edge_intensity": 0.6},
            "shape": {"shapes_detected": ["Square"]},
            "growth_rate": "normal",
            "thickness": "thick",
            "surface_changes": ["yellow discoloration"]
        },
        "outputs": {
            "disease": "Acne (Yauvan Pidika)",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "mildly imbalanced",
                "kapha": "slightly imbalanced"
            },
            "observations": [
                "Oily skin texture and acne indicate Pitta imbalance.",
                "Yellow discoloration on nails suggests fungal infection."
            ],
            "recommendations": {
                "diet": ["Include bitter foods like neem and karela.", "Avoid fried and spicy foods."],
                "lifestyle": ["Wash the face with a natural neem-based cleanser.", "Avoid touching or picking at acne."],
                "herbal_remedies": [
                    "Sandalwood paste and rose water for cooling.",
                    "Kalonji (black seed) oil for skin purification.",
                    "Pippali (long pepper) powder to reduce inflammation."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "light",
            "texture": "dry",
            "dark_circles": "present",
            "spots": [{"location": "x=35, y=180", "size": 3}]
        },
        "nail_analysis": {
            "color": {"RGB": [210, 180, 150], "Hex": "#d2b496"},
            "texture": {"edge_intensity": 0.3},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "slow",
            "thickness": "thin",
            "surface_changes": ["peeling"]
        },
        "outputs": {
            "disease": "Vitamin Deficiency",
            "dosha_analysis": {
                "vata": "mildly imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Peeling nails suggest vitamin deficiency or dryness.",
                "Dry skin with dark circles indicates dehydration."
            ],
            "recommendations": {
                "diet": ["Consume foods rich in vitamin E like almonds and sunflower seeds.", "Hydrate with coconut water."],
                "lifestyle": ["Massage nails with coconut oil daily.", "Avoid excessive exposure to harsh soaps."],
                "herbal_remedies": [
                    "Aloe vera gel for hydration.",
                    "Shankhapushpi for improving nutrient absorption.",
                    "Ashwagandha for balancing Vata and reducing dryness."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "medium",
            "texture": "normal",
            "dark_circles": "absent",
            "spots": [{"location": "x=25, y=200", "size": 4}]
        },
        "nail_analysis": {
            "color": {"RGB": [170, 140, 120], "Hex": "#aa8c78"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "normal",
            "thickness": "thick",
            "surface_changes": ["dark streaks"]
        },
        "outputs": {
            "disease": "Melanonychia (Dark Nail Streaks)",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "mildly imbalanced",
                "kapha": "balanced"
            },
            "observations": [
                "Dark streaks on nails indicate melanonychia.",
                "Normal skin texture suggests no major skin imbalance."
            ],
            "recommendations": {
                "diet": ["Consume antioxidant-rich foods like berries and pomegranate.", "Avoid greasy and fried foods."],
                "lifestyle": ["Keep nails moisturized with almond oil.", "Avoid trauma to the nail beds."],
                "herbal_remedies": [
                    "Turmeric for reducing inflammation.",
                    "Guggulu for improving blood circulation.",
                    "Saffron for enhancing skin and nail health."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "dark",
            "texture": "oily",
            "dark_circles": "present",
            "spots": [{"location": "x=45, y=190", "size": 5}]
        },
        "nail_analysis": {
            "color": {"RGB": [100, 80, 60], "Hex": "#64503c"},
            "texture": {"edge_intensity": 0.5},
            "shape": {"shapes_detected": ["Square"]},
            "growth_rate": "normal",
            "thickness": "thick",
            "surface_changes": ["yellow discoloration"]
        },
        "outputs": {
            "disease": "Fungal Nail Infection",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "mildly imbalanced",
                "kapha": "severely imbalanced"
            },
            "observations": [
                "Thick, discolored nails suggest fungal infection.",
                "Oily skin indicates Kapha imbalance."
            ],
            "recommendations": {
                "diet": ["Avoid sugary and processed foods.", "Include bitter foods like neem and turmeric."],
                "lifestyle": ["Keep nails dry and clean.", "Avoid tight-fitting shoes."],
                "herbal_remedies": [
                    "Neem oil for antifungal effects.",
                    "Moringa powder for boosting immunity.",
                    "Kalonji (black seed) oil for antifungal treatment."
                ]
            }
        }
    },
     {
        "skin_analysis": {
            "tone": "light",
            "texture": "smooth",
            "dark_circles": "absent",
            "spots": [{"location": "x=25, y=180", "size": 3}]
        },
        "nail_analysis": {
            "color": {"RGB": [200, 170, 150], "Hex": "#c8aa96"},
            "texture": {"edge_intensity": 0.2},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "normal",
            "thickness": "thin",
            "surface_changes": ["peeling"]
        },
        "outputs": {
            "disease": "Vitamin Deficiency",
            "dosha_analysis": {
                "vata": "mildly imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Peeling nails suggest vitamin deficiency or dryness.",
                "Smooth skin indicates no underlying skin issues."
            ],
            "recommendations": {
                "diet": ["Include foods rich in vitamin E like almonds and sunflower seeds.", "Consume papaya and leafy greens."],
                "lifestyle": ["Massage nails with coconut oil to restore moisture.", "Avoid harsh soaps and detergents."],
                "herbal_remedies": [
                    "Papaya pulp for hydration.",
                    "Hibiscus tea to improve nail strength.",
                    "Cardamom for better nutrient absorption."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "dark",
            "texture": "oily",
            "dark_circles": "present",
            "spots": [{"location": "x=40, y=200", "size": 5}]
        },
        "nail_analysis": {
            "color": {"RGB": [90, 70, 50], "Hex": "#5a4632"},
            "texture": {"edge_intensity": 0.5},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "normal",
            "thickness": "thick",
            "surface_changes": ["yellow discoloration"]
        },
        "outputs": {
            "disease": "Fungal Nail Infection",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "mildly imbalanced",
                "kapha": "imbalanced"
            },
            "observations": [
                "Thick, discolored nails suggest fungal infection.",
                "Oily skin indicates Kapha dominance."
            ],
            "recommendations": {
                "diet": ["Avoid sugary and processed foods.", "Include bitter foods like karela and neem."],
                "lifestyle": ["Keep nails dry and clean.", "Avoid wearing tight-fitting shoes."],
                "herbal_remedies": [
                    "Clove oil for antifungal application.",
                    "Kalonji oil to reduce fungal growth.",
                    "Vacha powder for detoxifying the body."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "medium",
            "texture": "rough",
            "dark_circles": "absent",
            "spots": [{"location": "x=35, y=170", "size": 4}]
        },
        "nail_analysis": {
            "color": {"RGB": [180, 150, 120], "Hex": "#b49678"},
            "texture": {"edge_intensity": 0.3},
            "shape": {"shapes_detected": ["Square"]},
            "growth_rate": "slow",
            "thickness": "thin",
            "surface_changes": ["ridges"]
        },
        "outputs": {
            "disease": "Age-related Brittle Nails",
            "dosha_analysis": {
                "vata": "severely imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Thin nails with ridges indicate aging-related dryness.",
                "Rough skin suggests Vata imbalance."
            ],
            "recommendations": {
                "diet": ["Consume warm, nourishing foods like ghee and soups.", "Avoid cold and dry foods."],
                "lifestyle": ["Massage nails with sesame oil daily.", "Avoid excessive exposure to water and detergents."],
                "herbal_remedies": [
                    "Musta powder to balance Vata.",
                    "Brahmi for improving nail health.",
                    "Shankhapushpi for hydration."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "light",
            "texture": "oily",
            "dark_circles": "present",
            "spots": [{"location": "x=20, y=150", "size": 3}]
        },
        "nail_analysis": {
            "color": {"RGB": [210, 190, 160], "Hex": "#d2bea0"},
            "texture": {"edge_intensity": 0.2},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "normal",
            "thickness": "thick",
            "surface_changes": ["dark streaks"]
        },
        "outputs": {
            "disease": "Melanonychia (Dark Nail Streaks)",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "mildly imbalanced",
                "kapha": "balanced"
            },
            "observations": [
                "Dark streaks on nails indicate melanonychia.",
                "Oily skin indicates no dryness-related issues."
            ],
            "recommendations": {
                "diet": ["Consume antioxidant-rich foods like berries and pomegranate.", "Include curry leaves in meals."],
                "lifestyle": ["Keep nails moisturized with almond oil.", "Avoid trauma to the nail beds."],
                "herbal_remedies": [
                    "Turmeric to reduce pigmentation.",
                    "Kanchanar for improving skin and nail health.",
                    "Bitter orange for detoxification."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "dark",
            "texture": "dry",
            "dark_circles": "absent",
            "spots": [{"location": "x=50, y=200", "size": 6}]
        },
        "nail_analysis": {
            "color": {"RGB": [100, 80, 60], "Hex": "#64503c"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "slow",
            "thickness": "thin",
            "surface_changes": ["peeling"]
        },
        "outputs": {
            "disease": "Dehydration-related Nail Peeling",
            "dosha_analysis": {
                "vata": "severely imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Peeling nails suggest dehydration and dryness.",
                "Dry skin indicates severe Vata imbalance."
            ],
            "recommendations": {
                "diet": ["Hydrate with plenty of water and coconut water.", "Consume foods like cucumber and fennel."],
                "lifestyle": ["Massage nails with coconut oil daily.", "Avoid harsh chemicals and soaps."],
                "herbal_remedies": [
                    "Fennel seeds for cooling the body.",
                    "Kushta for reducing dryness.",
                    "Licorice for hydration."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "medium",
            "texture": "normal",
            "dark_circles": "absent",
            "spots": [{"location": "x=30, y=180", "size": 4}]
        },
        "nail_analysis": {
            "color": {"RGB": [170, 140, 120], "Hex": "#aa8c78"},
            "texture": {"edge_intensity": 0.3},
            "shape": {"shapes_detected": ["Square"]},
            "growth_rate": "normal",
            "thickness": "thin",
            "surface_changes": ["white spots"]
        },
        "outputs": {
            "disease": "Leukonychia (White Spots on Nails)",
            "dosha_analysis": {
                "vata": "slightly imbalanced",
                "pitta": "balanced",
                "kapha": "slightly imbalanced"
            },
            "observations": [
                "White spots on nails suggest trauma or zinc deficiency.",
                "Normal skin texture suggests no severe skin imbalance."
            ],
            "recommendations": {
                "diet": ["Include zinc-rich foods like pumpkin seeds and lentils.", "Consume sesame seeds to improve nail health."],
                "lifestyle": ["Avoid biting nails or exposing them to harsh chemicals.", "Massage nails with coconut oil to restore shine."],
                "herbal_remedies": [
                    "Rasna for improving nail strength.",
                    "Aloe vera juice for internal hydration.",
                    "Vacha for detoxification."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "medium",
            "texture": "dry",
            "dark_circles": "present",
            "spots": [{"location": "x=30, y=160", "size": 4}]
        },
        "nail_analysis": {
            "color": {"RGB": [180, 150, 120], "Hex": "#b49678"},
            "texture": {"edge_intensity": 0.3},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "slow",
            "thickness": "thin",
            "surface_changes": ["longitudinal ridges"]
        },
        "outputs": {
            "disease": "Eczema (Vicharchika)",
            "dosha_analysis": {
                "vata": "severely imbalanced",
                "pitta": "balanced",
                "kapha": "slightly imbalanced"
            },
            "observations": [
                "Dry, itchy, and inflamed skin indicates severe Vata imbalance.",
                "Longitudinal ridges on nails suggest aging-related changes."
            ],
            "recommendations": {
                "diet": ["Consume warm, moist foods like soups and stews.", "Avoid cold, dry, and processed foods."],
                "lifestyle": ["Apply Khadira paste to inflamed areas.", "Avoid harsh soaps and detergents."],
                "herbal_remedies": [
                    "Kushta for reducing dryness and inflammation.",
                    "Vacha for detoxification.",
                    "Coriander seeds for cooling the body."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "dark",
            "texture": "oily",
            "dark_circles": "absent",
            "spots": [{"location": "x=40, y=200", "size": 5}]
        },
        "nail_analysis": {
            "color": {"RGB": [90, 70, 50], "Hex": "#5a4632"},
            "texture": {"edge_intensity": 0.5},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "normal",
            "thickness": "thick",
            "surface_changes": ["yellow discoloration"]
        },
        "outputs": {
            "disease": "Fungal Nail Infection",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "mildly imbalanced",
                "kapha": "imbalanced"
            },
            "observations": [
                "Thick, discolored nails suggest fungal infection.",
                "Oily skin indicates Kapha dominance."
            ],
            "recommendations": {
                "diet": ["Avoid sugary and processed foods.", "Include bitter foods like karela and neem."],
                "lifestyle": ["Keep nails dry and clean.", "Avoid wearing tight-fitting shoes."],
                "herbal_remedies": [
                    "Clove oil for antifungal application.",
                    "Kalonji oil to reduce fungal growth.",
                    "Rasna for detoxifying the body."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "light",
            "texture": "normal",
            "dark_circles": "present",
            "spots": [{"location": "x=20, y=150", "size": 3}]
        },
        "nail_analysis": {
            "color": {"RGB": [210, 190, 160], "Hex": "#d2bea0"},
            "texture": {"edge_intensity": 0.2},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "normal",
            "thickness": "thin",
            "surface_changes": ["dark streaks"]
        },
        "outputs": {
            "disease": "Melanonychia (Dark Nail Streaks)",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "mildly imbalanced",
                "kapha": "balanced"
            },
            "observations": [
                "Dark streaks on nails indicate melanonychia.",
                "Normal skin texture suggests no dryness-related issues."
            ],
            "recommendations": {
                "diet": ["Consume antioxidant-rich foods like berries and pomegranate.", "Include curry leaves in meals."],
                "lifestyle": ["Keep nails moisturized with almond oil.", "Avoid trauma to the nail beds."],
                "herbal_remedies": [
                    "Turmeric to reduce pigmentation.",
                    "Kanchanar for improving skin and nail health.",
                    "Bitter orange for detoxification."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "medium",
            "texture": "rough",
            "dark_circles": "absent",
            "spots": [{"location": "x=35, y=170", "size": 4}]
        },
        "nail_analysis": {
            "color": {"RGB": [180, 150, 120], "Hex": "#b49678"},
            "texture": {"edge_intensity": 0.3},
            "shape": {"shapes_detected": ["Square"]},
            "growth_rate": "slow",
            "thickness": "thin",
            "surface_changes": ["ridges"]
        },
        "outputs": {
            "disease": "Age-related Brittle Nails",
            "dosha_analysis": {
                "vata": "severely imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Thin nails with ridges indicate aging-related dryness.",
                "Rough skin suggests Vata imbalance."
            ],
            "recommendations": {
                "diet": ["Consume warm, nourishing foods like ghee and soups.", "Avoid cold and dry foods."],
                "lifestyle": ["Massage nails with sesame oil daily.", "Avoid excessive exposure to water and detergents."],
                "herbal_remedies": [
                    "Musta powder to balance Vata.",
                    "Brahmi for improving nail health.",
                    "Shankhapushpi for hydration."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "dark",
            "texture": "dry",
            "dark_circles": "absent",
            "spots": [{"location": "x=50, y=200", "size": 6}]
        },
        "nail_analysis": {
            "color": {"RGB": [100, 80, 60], "Hex": "#64503c"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "slow",
            "thickness": "thin",
            "surface_changes": ["peeling"]
        },
        "outputs": {
            "disease": "Dehydration-related Nail Peeling",
            "dosha_analysis": {
                "vata": "severely imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Peeling nails suggest dehydration and dryness.",
                "Dry skin indicates severe Vata imbalance."
            ],
            "recommendations": {
                "diet": ["Hydrate with plenty of water and coconut water.", "Consume foods like cucumber and fennel."],
                "lifestyle": ["Massage nails with coconut oil daily.", "Avoid harsh chemicals and soaps."],
                "herbal_remedies": [
                    "Fennel seeds for cooling the body.",
                    "Kushta for reducing dryness.",
                    "Licorice for hydration."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "light",
            "texture": "oily",
            "dark_circles": "present",
            "spots": [{"location": "x=25, y=180", "size": 3}]
        },
        "nail_analysis": {
            "color": {"RGB": [200, 170, 150], "Hex": "#c8aa96"},
            "texture": {"edge_intensity": 0.2},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "normal",
            "thickness": "thin",
            "surface_changes": ["peeling"]
        },
        "outputs": {
            "disease": "Vitamin Deficiency",
            "dosha_analysis": {
                "vata": "mildly imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Peeling nails suggest vitamin deficiency or dryness.",
                "Smooth skin indicates no underlying skin issues."
            ],
            "recommendations": {
                "diet": ["Include foods rich in vitamin E like almonds and sunflower seeds.", "Consume papaya and leafy greens."],
                "lifestyle": ["Massage nails with coconut oil to restore moisture.", "Avoid harsh soaps and detergents."],
                "herbal_remedies": [
                    "Papaya pulp for hydration.",
                    "Hibiscus tea to improve nail strength.",
                    "Cardamom for better nutrient absorption."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "dark",
            "texture": "oily",
            "dark_circles": "present",
            "spots": [{"location": "x=30, y=200", "size": 4}]
        },
        "nail_analysis": {
            "color": {"RGB": [90, 70, 50], "Hex": "#5a4632"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "normal",
            "thickness": "thick",
            "surface_changes": ["yellow discoloration"]
        },
        "outputs": {
            "disease": "Fungal Infection",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "mildly imbalanced",
                "kapha": "severely imbalanced"
            },
            "observations": [
                "Thick, discolored nails suggest fungal infection.",
                "Oily skin indicates Kapha imbalance."
            ],
            "recommendations": {
                "diet": ["Avoid sugary and processed foods.", "Include bitter foods like karela and neem."],
                "lifestyle": ["Keep nails dry and clean.", "Avoid tight-fitting shoes and synthetic fabrics."],
                "herbal_remedies": [
                    "Kalonji oil for antifungal effects.",
                    "Punarnava to reduce swelling and detoxify.",
                    "Clove oil for topical application."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "medium",
            "texture": "dry",
            "dark_circles": "absent",
            "spots": [{"location": "x=40, y=180", "size": 3}]
        },
        "nail_analysis": {
            "color": {"RGB": [180, 150, 120], "Hex": "#b49678"},
            "texture": {"edge_intensity": 0.3},
            "shape": {"shapes_detected": ["Square"]},
            "growth_rate": "slow",
            "thickness": "thin",
            "surface_changes": ["longitudinal ridges"]
        },
        "outputs": {
            "disease": "Age-related Brittle Nails",
            "dosha_analysis": {
                "vata": "severely imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Thin nails with ridges indicate aging-related Vata imbalance.",
                "Dry skin suggests dehydration and Vata dominance."
            ],
            "recommendations": {
                "diet": ["Consume warm, nourishing foods like ghee and soups.", "Avoid cold and dry foods."],
                "lifestyle": ["Massage nails with sesame oil daily.", "Avoid repeated exposure to harsh soaps and detergents."],
                "herbal_remedies": [
                    "Vacha to improve digestion and balance Vata.",
                    "Musta for hydration and skin health.",
                    "Kushta for reducing inflammation and dryness."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "light",
            "texture": "smooth",
            "dark_circles": "present",
            "spots": [{"location": "x=25, y=150", "size": 2}]
        },
        "nail_analysis": {
            "color": {"RGB": [210, 190, 160], "Hex": "#d2bea0"},
            "texture": {"edge_intensity": 0.2},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "normal",
            "thickness": "thin",
            "surface_changes": ["peeling"]
        },
        "outputs": {
            "disease": "Vitamin Deficiency",
            "dosha_analysis": {
                "vata": "mildly imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Peeling nails suggest vitamin deficiency or dryness.",
                "Smooth skin indicates no severe external issues."
            ],
            "recommendations": {
                "diet": ["Consume foods rich in vitamin E like almonds and sunflower seeds.", "Include papaya and leafy greens in your diet."],
                "lifestyle": ["Massage nails with coconut oil daily to restore moisture.", "Avoid exposure to harsh soaps."],
                "herbal_remedies": [
                    "Papaya pulp for hydration.",
                    "Aloe vera gel for soothing nail beds.",
                    "Gotu Kola for improving skin and nail health."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "dark",
            "texture": "oily",
            "dark_circles": "present",
            "spots": [{"location": "x=50, y=200", "size": 6}]
        },
        "nail_analysis": {
            "color": {"RGB": [100, 80, 60], "Hex": "#64503c"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "normal",
            "thickness": "thick",
            "surface_changes": ["yellow discoloration"]
        },
        "outputs": {
            "disease": "Fungal Nail Infection",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "slightly imbalanced",
                "kapha": "severely imbalanced"
            },
            "observations": [
                "Thick, yellow nails suggest fungal infection.",
                "Oily skin indicates Kapha imbalance."
            ],
            "recommendations": {
                "diet": ["Avoid sugar and processed foods.", "Consume bitter melon and neem leaves."],
                "lifestyle": ["Keep nails trimmed and dry.", "Avoid tight footwear to minimize fungal growth."],
                "herbal_remedies": [
                    "Kanchanar to detoxify.",
                    "Chirata to support liver health and immunity.",
                    "Clove oil for topical antifungal use."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "medium",
            "texture": "normal",
            "dark_circles": "present",
            "spots": [{"location": "x=35, y=180", "size": 4}]
        },
        "nail_analysis": {
            "color": {"RGB": [180, 150, 130], "Hex": "#b49682"},
            "texture": {"edge_intensity": 0.3},
            "shape": {"shapes_detected": ["Square"]},
            "growth_rate": "slow",
            "thickness": "thin",
            "surface_changes": ["white spots"]
        },
        "outputs": {
            "disease": "Leukonychia (White Spots on Nails)",
            "dosha_analysis": {
                "vata": "slightly imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "White spots on nails suggest trauma or zinc deficiency.",
                "Dark circles indicate mild dehydration."
            ],
            "recommendations": {
                "diet": ["Include zinc-rich foods like pumpkin seeds and lentils.", "Consume fresh coriander for detoxification."],
                "lifestyle": ["Avoid biting nails and harsh chemicals.", "Massage nails with almond oil."],
                "herbal_remedies": [
                    "Vidanga for detoxifying the body.",
                    "Punarnava for improving hydration.",
                    "Sarpagandha for calming stress-induced imbalances."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "light",
            "texture": "dry",
            "dark_circles": "absent",
            "spots": [{"location": "x=20, y=150", "size": 3}]
        },
        "nail_analysis": {
            "color": {"RGB": [210, 180, 140], "Hex": "#d2b48c"},
            "texture": {"edge_intensity": 0.2},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "normal",
            "thickness": "thin",
            "surface_changes": ["peeling"]
        },
        "outputs": {
            "disease": "Dehydration-related Nail Peeling",
            "dosha_analysis": {
                "vata": "severely imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Peeling nails suggest dehydration and dryness.",
                "Dry skin indicates severe Vata imbalance."
            ],
            "recommendations": {
                "diet": ["Hydrate with plenty of water and fresh coconut water.", "Consume fennel seeds to cool the body."],
                "lifestyle": ["Use aloe vera gel to massage nails and restore hydration.", "Avoid exposure to excessive soaps and detergents."],
                "herbal_remedies": [
                    "Kali Musli for improving hydration and vitality.",
                    "Vacha for improving digestion and reducing dryness.",
                    "Sarpagandha for reducing stress and balancing Vata."
                ]
            }
        }
    },
     {
        "skin_analysis": {
            "tone": "light",
            "texture": "smooth",
            "dark_circles": "present",
            "spots": [{"location": "x=25, y=155", "size": 3}]
        },
        "nail_analysis": {
            "color": {"RGB": [210, 190, 160], "Hex": "#d2bea0"},
            "texture": {"edge_intensity": 0.3},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "normal",
            "thickness": "thin",
            "surface_changes": ["peeling"]
        },
        "outputs": {
            "disease": "Vitamin Deficiency",
            "dosha_analysis": {
                "vata": "mildly imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Peeling nails suggest vitamin deficiency or dryness.",
                "Smooth skin indicates no severe external issues."
            ],
            "recommendations": {
                "diet": ["Include vitamin-rich foods like spinach and almonds.", "Consume fresh papaya to improve hydration."],
                "lifestyle": ["Massage nails with aloe vera gel daily.", "Avoid harsh detergents to prevent dryness."],
                "herbal_remedies": [
                    "Gotu Kola for skin and nail health.",
                    "Punarnava to improve hydration and reduce dryness.",
                    "Kali Musli for improving vitality."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "medium",
            "texture": "rough",
            "dark_circles": "absent",
            "spots": [{"location": "x=30, y=175", "size": 5}]
        },
        "nail_analysis": {
            "color": {"RGB": [180, 150, 120], "Hex": "#b49678"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "slow",
            "thickness": "thin",
            "surface_changes": ["longitudinal ridges"]
        },
        "outputs": {
            "disease": "Age-related Brittle Nails",
            "dosha_analysis": {
                "vata": "severely imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Thin nails with ridges indicate aging-related dryness.",
                "Rough skin suggests Vata dominance."
            ],
            "recommendations": {
                "diet": ["Consume warm, nourishing foods like soups and ghee.", "Avoid cold and dry foods."],
                "lifestyle": ["Massage nails with sesame oil daily.", "Avoid prolonged exposure to water."],
                "herbal_remedies": [
                    "Boswellia for reducing inflammation.",
                    "Chirata for improving liver function and hydration.",
                    "Musta for balancing Vata."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "dark",
            "texture": "oily",
            "dark_circles": "present",
            "spots": [{"location": "x=40, y=190", "size": 4}]
        },
        "nail_analysis": {
            "color": {"RGB": [90, 70, 50], "Hex": "#5a4632"},
            "texture": {"edge_intensity": 0.5},
            "shape": {"shapes_detected": ["Square"]},
            "growth_rate": "normal",
            "thickness": "thick",
            "surface_changes": ["yellow discoloration"]
        },
        "outputs": {
            "disease": "Fungal Nail Infection",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "mildly imbalanced",
                "kapha": "severely imbalanced"
            },
            "observations": [
                "Thick, discolored nails suggest fungal infection.",
                "Oily skin indicates Kapha imbalance."
            ],
            "recommendations": {
                "diet": ["Include bitter foods like karela and neem leaves.", "Avoid sugary foods to reduce fungal growth."],
                "lifestyle": ["Keep nails dry and clean.", "Use breathable footwear to prevent excess moisture."],
                "herbal_remedies": [
                    "Vidanga for its anti-parasitic properties.",
                    "Nirgundi oil for reducing inflammation.",
                    "Clove oil for antifungal application."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "medium",
            "texture": "oily",
            "dark_circles": "absent",
            "spots": [{"location": "x=50, y=200", "size": 6}]
        },
        "nail_analysis": {
            "color": {"RGB": [170, 140, 120], "Hex": "#aa8c78"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "normal",
            "thickness": "thick",
            "surface_changes": ["dark streaks"]
        },
        "outputs": {
            "disease": "Melanonychia (Dark Nail Streaks)",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "mildly imbalanced",
                "kapha": "slightly imbalanced"
            },
            "observations": [
                "Dark streaks on nails suggest pigmentation issues.",
                "Oily skin texture indicates mild Kapha imbalance."
            ],
            "recommendations": {
                "diet": ["Consume antioxidant-rich foods like berries and pomegranate.", "Include coriander to detoxify the body."],
                "lifestyle": ["Keep nails moisturized with almond oil.", "Avoid excessive sun exposure to prevent hyperpigmentation."],
                "herbal_remedies": [
                    "Sarpagandha for calming stress-related imbalances.",
                    "Arjuna for improving circulation.",
                    "Ashoka for detoxification and skin health."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "light",
            "texture": "normal",
            "dark_circles": "present",
            "spots": [{"location": "x=20, y=150", "size": 2}]
        },
        "nail_analysis": {
            "color": {"RGB": [210, 180, 140], "Hex": "#d2b48c"},
            "texture": {"edge_intensity": 0.2},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "normal",
            "thickness": "thin",
            "surface_changes": ["peeling"]
        },
        "outputs": {
            "disease": "Dehydration-related Nail Peeling",
            "dosha_analysis": {
                "vata": "severely imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Peeling nails suggest dehydration and dryness.",
                "Dark circles indicate mild Vata imbalance."
            ],
            "recommendations": {
                "diet": ["Hydrate with plenty of water and fresh coconut water.", "Consume fennel seeds to cool the body."],
                "lifestyle": ["Massage nails with aloe vera gel daily.", "Avoid excessive soap usage to prevent further dryness."],
                "herbal_remedies": [
                    "Kesar for improving hydration and mood.",
                    "Moringa for boosting overall health.",
                    "Gotu Kola for improving nail and skin strength."
                ]
            }
        }
    },
{
        "skin_analysis": {
            "tone": "medium",
            "texture": "rough",
            "dark_circles": "absent",
            "spots": [{"location": "x=30, y=180", "size": 4}]
        },
        "nail_analysis": {
            "color": {"RGB": [180, 150, 120], "Hex": "#b49678"},
            "texture": {"edge_intensity": 0.5},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "slow",
            "thickness": "thin",
            "surface_changes": ["longitudinal ridges"]
        },
        "outputs": {
            "disease": "Rheumatoid Arthritis",
            "dosha_analysis": {
                "vata": "severely imbalanced",
                "pitta": "mildly imbalanced",
                "kapha": "balanced"
            },
            "observations": [
                "Longitudinal ridges on nails suggest systemic inflammation.",
                "Rough skin texture indicates Vata imbalance."
            ],
            "recommendations": {
                "diet": ["Consume anti-inflammatory foods like turmeric and ginger.", "Avoid cold and dry foods."],
                "lifestyle": ["Regular oil massage for joints and nails.", "Practice gentle yoga exercises to improve mobility."],
                "herbal_remedies": [
                    "Boswellia for reducing joint inflammation.",
                    "Punarnava for systemic detoxification.",
                    "Guggulu for joint health."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "dark",
            "texture": "oily",
            "dark_circles": "present",
            "spots": [{"location": "x=40, y=200", "size": 5}]
        },
        "nail_analysis": {
            "color": {"RGB": [90, 70, 50], "Hex": "#5a4632"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Square"]},
            "growth_rate": "normal",
            "thickness": "thick",
            "surface_changes": ["yellow discoloration"]
        },
        "outputs": {
            "disease": "Psoriasis",
            "dosha_analysis": {
                "vata": "mildly imbalanced",
                "pitta": "severely imbalanced",
                "kapha": "imbalanced"
            },
            "observations": [
                "Yellow discoloration on nails indicates psoriatic involvement.",
                "Oily skin suggests Kapha imbalance."
            ],
            "recommendations": {
                "diet": ["Avoid spicy and fried foods.", "Consume cooling foods like cucumber and coriander."],
                "lifestyle": ["Use neem-based skincare products.", "Avoid stress and harsh detergents."],
                "herbal_remedies": [
                    "Khadira for skin purification.",
                    "Neem capsules to reduce inflammation.",
                    "Bitter melon for detoxification."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "light",
            "texture": "smooth",
            "dark_circles": "present",
            "spots": [{"location": "x=25, y=160", "size": 2}]
        },
        "nail_analysis": {
            "color": {"RGB": [210, 190, 160], "Hex": "#d2bea0"},
            "texture": {"edge_intensity": 0.3},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "normal",
            "thickness": "thin",
            "surface_changes": ["peeling"]
        },
        "outputs": {
            "disease": "Vitamin B12 Deficiency",
            "dosha_analysis": {
                "vata": "mildly imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Peeling nails suggest vitamin deficiency.",
                "Smooth skin indicates no external inflammation."
            ],
            "recommendations": {
                "diet": ["Consume vitamin B12-rich foods like dairy and eggs.", "Include fresh leafy greens in daily meals."],
                "lifestyle": ["Massage nails with coconut oil to restore hydration.", "Avoid excessive water exposure."],
                "herbal_remedies": [
                    "Kali Musli for improving vitality.",
                    "Punarnava for hydration and detoxification.",
                    "Gotu Kola for skin and cognitive health."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "medium",
            "texture": "dry",
            "dark_circles": "absent",
            "spots": [{"location": "x=35, y=170", "size": 4}]
        },
        "nail_analysis": {
            "color": {"RGB": [180, 160, 140], "Hex": "#b4a08c"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Square"]},
            "growth_rate": "slow",
            "thickness": "thin",
            "surface_changes": ["longitudinal ridges"]
        },
        "outputs": {
            "disease": "Hypothyroidism",
            "dosha_analysis": {
                "vata": "imbalanced",
                "pitta": "balanced",
                "kapha": "severely imbalanced"
            },
            "observations": [
                "Slow nail growth and longitudinal ridges suggest hypothyroidism.",
                "Dry skin indicates Kapha imbalance."
            ],
            "recommendations": {
                "diet": ["Consume iodine-rich foods like seaweed and fish.", "Avoid processed and refined foods."],
                "lifestyle": ["Regular physical exercise to improve metabolism.", "Use warm sesame oil for skin massage."],
                "herbal_remedies": [
                    "Kanchanar for thyroid support.",
                    "Ashoka for balancing hormones.",
                    "Triphala for detoxification."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "dark",
            "texture": "normal",
            "dark_circles": "present",
            "spots": [{"location": "x=30, y=180", "size": 3}]
        },
        "nail_analysis": {
            "color": {"RGB": [90, 70, 50], "Hex": "#5a4632"},
            "texture": {"edge_intensity": 0.3},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "normal",
            "thickness": "thick",
            "surface_changes": ["white spots"]
        },
        "outputs": {
            "disease": "Celiac Disease",
            "dosha_analysis": {
                "vata": "slightly imbalanced",
                "pitta": "imbalanced",
                "kapha": "balanced"
            },
            "observations": [
                "White spots on nails indicate nutritional malabsorption.",
                "Dark circles suggest fatigue and digestive issues."
            ],
            "recommendations": {
                "diet": ["Completely avoid gluten-containing foods.", "Consume fresh fruits and vegetables."],
                "lifestyle": ["Practice mindful eating to improve digestion.", "Stay hydrated with warm water throughout the day."],
                "herbal_remedies": [
                    "Vidanga to support digestion.",
                    "Chirata for improving liver health.",
                    "Musta for alleviating gastrointestinal discomfort."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "light",
            "texture": "normal",
            "dark_circles": "absent",
            "spots": [{"location": "x=20, y=150", "size": 2}]
        },
        "nail_analysis": {
            "color": {"RGB": [210, 190, 150], "Hex": "#d2be96"},
            "texture": {"edge_intensity": 0.2},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "normal",
            "thickness": "thin",
            "surface_changes": ["peeling"]
        },
        "outputs": {
            "disease": "Vitamin C Deficiency (Scurvy)",
            "dosha_analysis": {
                "vata": "imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Peeling nails suggest vitamin C deficiency.",
                "Normal skin suggests no severe external inflammation."
            ],
            "recommendations": {
                "diet": ["Consume vitamin C-rich foods like citrus fruits and amla.", "Include fresh vegetables in meals."],
                "lifestyle": ["Avoid overly processed and canned foods.", "Practice regular hydration to improve skin and nails."],
                "herbal_remedies": [
                    "Amla for boosting vitamin C levels.",
                    "Neem for purifying blood.",
                    "Triphala for supporting digestion and nutrient absorption."
                ]
            }
        }
    },
     {
        "skin_analysis": {
            "tone": "light",
            "texture": "smooth",
            "dark_circles": "present",
            "spots": [{"location": "x=25, y=150", "size": 3}]
        },
        "nail_analysis": {
            "color": {"RGB": [210, 190, 160], "Hex": "#d2bea0"},
            "texture": {"edge_intensity": 0.2},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "normal",
            "thickness": "thin",
            "surface_changes": ["peeling"]
        },
        "outputs": {
            "disease": "Vitamin B12 Deficiency",
            "dosha_analysis": {
                "vata": "mildly imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Peeling nails suggest vitamin deficiency.",
                "Smooth skin indicates no severe external issues."
            ],
            "recommendations": {
                "diet": ["Consume foods rich in vitamin B12 like fish, meat, and eggs."],
                "lifestyle": ["Massage nails with coconut oil daily."],
                "herbal_remedies": [
                    "Ashwagandha for improving energy levels.",
                    "Gotu Kola for improving skin and nail health."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "medium",
            "texture": "dry",
            "dark_circles": "absent",
            "spots": [{"location": "x=30, y=180", "size": 5}]
        },
        "nail_analysis": {
            "color": {"RGB": [180, 150, 120], "Hex": "#b49678"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "slow",
            "thickness": "thin",
            "surface_changes": ["longitudinal ridges"]
        },
        "outputs": {
            "disease": "Iron Deficiency",
            "dosha_analysis": {
                "vata": "severely imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Thin nails with ridges indicate anemia caused by iron deficiency.",
                "Dry skin suggests dehydration and Vata imbalance."
            ],
            "recommendations": {
                "diet": ["Include iron-rich foods like spinach, lentils, and red meat."],
                "lifestyle": ["Avoid excessive caffeine, which can block iron absorption."],
                "herbal_remedies": [
                    "Punarnava for supporting iron levels.",
                    "Khadira for improving skin and nail health."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "dark",
            "texture": "oily",
            "dark_circles": "present",
            "spots": [{"location": "x=40, y=200", "size": 4}]
        },
        "nail_analysis": {
            "color": {"RGB": [90, 70, 50], "Hex": "#5a4632"},
            "texture": {"edge_intensity": 0.5},
            "shape": {"shapes_detected": ["Square"]},
            "growth_rate": "normal",
            "thickness": "thick",
            "surface_changes": ["yellow discoloration"]
        },
        "outputs": {
            "disease": "Fungal Nail Infection",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "slightly imbalanced",
                "kapha": "severely imbalanced"
            },
            "observations": [
                "Thick, discolored nails suggest fungal infection.",
                "Dark circles indicate mild pitta imbalance."
            ],
            "recommendations": {
                "diet": ["Include bitter foods like karela and neem leaves."],
                "lifestyle": ["Keep nails dry and clean."],
                "herbal_remedies": [
                    "Vidanga for antifungal properties.",
                    "Clove oil for topical application."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "medium",
            "texture": "oily",
            "dark_circles": "absent",
            "spots": [{"location": "x=50, y=220", "size": 6}]
        },
        "nail_analysis": {
            "color": {"RGB": [170, 140, 120], "Hex": "#aa8c78"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "normal",
            "thickness": "thick",
            "surface_changes": ["dark streaks"]
        },
        "outputs": {
            "disease": "Melanonychia (Dark Nail Streaks)",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "mildly imbalanced",
                "kapha": "slightly imbalanced"
            },
            "observations": [
                "Dark streaks on nails suggest pigmentation issues.",
                "Oily skin indicates mild Kapha dominance."
            ],
            "recommendations": {
                "diet": ["Consume antioxidant-rich foods like blueberries and turmeric."],
                "lifestyle": ["Avoid harsh chemicals on nails."],
                "herbal_remedies": [
                    "Arjuna for improving circulation.",
                    "Ashoka for balancing pigmentation."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "light",
            "texture": "dry",
            "dark_circles": "absent",
            "spots": [{"location": "x=20, y=150", "size": 2}]
        },
        "nail_analysis": {
            "color": {"RGB": [210, 180, 140], "Hex": "#d2b48c"},
            "texture": {"edge_intensity": 0.2},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "normal",
            "thickness": "thin",
            "surface_changes": ["peeling"]
        },
        "outputs": {
            "disease": "Addison's Disease",
            "dosha_analysis": {
                "vata": "severely imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Peeling nails suggest dehydration and adrenal gland dysfunction.",
                "Dry skin indicates severe Vata imbalance."
            ],
            "recommendations": {
                "diet": ["Hydrate with plenty of water and coconut water."],
                "lifestyle": ["Avoid excessive physical and mental stress."],
                "herbal_remedies": [
                    "Kali Musli to improve adrenal function.",
                    "Punarnava for hydrating tissues."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "medium",
            "texture": "smooth",
            "dark_circles": "present",
            "spots": [{"location": "x=30, y=180", "size": 4}]
        },
        "nail_analysis": {
            "color": {"RGB": [180, 160, 130], "Hex": "#b49c82"},
            "texture": {"edge_intensity": 0.3},
            "shape": {"shapes_detected": ["Square"]},
            "growth_rate": "normal",
            "thickness": "thin",
            "surface_changes": ["white spots"]
        },
        "outputs": {
            "disease": "Leukonychia (White Spots on Nails)",
            "dosha_analysis": {
                "vata": "slightly imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "White spots on nails suggest trauma or zinc deficiency.",
                "Smooth skin indicates no severe external skin issues."
            ],
            "recommendations": {
                "diet": ["Include zinc-rich foods like pumpkin seeds."],
                "lifestyle": ["Avoid biting nails or exposing them to harsh chemicals."],
                "herbal_remedies": [
                    "Chirata for detoxifying the body.",
                    "Khadira for cleansing the skin."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "light",
            "texture": "dry",
            "dark_circles": "present",
            "spots": [{"location": "x=30, y=150", "size": 3}]
        },
        "nail_analysis": {
            "color": {"RGB": [210, 180, 150], "Hex": "#d2b496"},
            "texture": {"edge_intensity": 0.3},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "slow",
            "thickness": "thin",
            "surface_changes": ["peeling"]
        },
        "outputs": {
            "disease": "Vitamin B12 Deficiency",
            "dosha_analysis": {
                "vata": "mildly imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Peeling nails and dark circles suggest vitamin B12 deficiency.",
                "Dry skin indicates underlying nutrient depletion."
            ],
            "recommendations": {
                "diet": ["Include foods rich in vitamin B12 like dairy products and fortified cereals.", "Consume coriander and ghee for better absorption."],
                "lifestyle": ["Massage nails with almond oil daily.", "Avoid prolonged exposure to cold environments."],
                "herbal_remedies": [
                    "Ashoka for improving blood health.",
                    "Gotu Kola for cognitive function and skin repair.",
                    "Musta to aid digestion and enhance nutrient absorption."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "medium",
            "texture": "oily",
            "dark_circles": "present",
            "spots": [{"location": "x=40, y=200", "size": 5}]
        },
        "nail_analysis": {
            "color": {"RGB": [150, 130, 110], "Hex": "#967c6e"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "normal",
            "thickness": "thick",
            "surface_changes": ["yellow discoloration"]
        },
        "outputs": {
            "disease": "Celiac Disease",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "mildly imbalanced",
                "kapha": "slightly imbalanced"
            },
            "observations": [
                "Yellowing nails and oily skin indicate mild Kapha imbalance.",
                "Dark circles suggest nutrient malabsorption due to gluten sensitivity."
            ],
            "recommendations": {
                "diet": ["Avoid gluten-containing foods like wheat and barley.", "Include bitter melon for its digestive benefits."],
                "lifestyle": ["Drink warm water with ginger in the morning.", "Practice mindfulness to reduce stress on digestion."],
                "herbal_remedies": [
                    "Punarnava for improving gut health.",
                    "Vidanga to detoxify and cleanse the intestines.",
                    "Karela (bitter melon) to support digestion."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "dark",
            "texture": "rough",
            "dark_circles": "absent",
            "spots": [{"location": "x=35, y=180", "size": 4}]
        },
        "nail_analysis": {
            "color": {"RGB": [90, 70, 50], "Hex": "#5a4632"},
            "texture": {"edge_intensity": 0.5},
            "shape": {"shapes_detected": ["Square"]},
            "growth_rate": "slow",
            "thickness": "thin",
            "surface_changes": ["longitudinal ridges"]
        },
        "outputs": {
            "disease": "Rheumatoid Arthritis",
            "dosha_analysis": {
                "vata": "severely imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Rough skin and nail ridges suggest systemic Vata imbalance.",
                "Joint pain and stiffness are common symptoms."
            ],
            "recommendations": {
                "diet": ["Consume warm, nourishing foods like soups and stews.", "Avoid cold and dry foods."],
                "lifestyle": ["Apply warm sesame oil to joints for pain relief.", "Practice yoga to improve joint flexibility."],
                "herbal_remedies": [
                    "Boswellia for its anti-inflammatory properties.",
                    "Nirgundi oil to reduce joint swelling.",
                    "Ashoka for balancing Vata and reducing stiffness."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "medium",
            "texture": "oily",
            "dark_circles": "present",
            "spots": [{"location": "x=50, y=220", "size": 6}]
        },
        "nail_analysis": {
            "color": {"RGB": [170, 140, 120], "Hex": "#aa8c78"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "normal",
            "thickness": "thick",
            "surface_changes": ["yellow discoloration"]
        },
        "outputs": {
            "disease": "Type 1 Diabetes",
            "dosha_analysis": {
                "vata": "mildly imbalanced",
                "pitta": "mildly imbalanced",
                "kapha": "balanced"
            },
            "observations": [
                "Oily skin and yellow nails suggest mild metabolic imbalance.",
                "Dark circles indicate blood sugar irregularities."
            ],
            "recommendations": {
                "diet": ["Avoid sugary and processed foods.", "Include fenugreek seeds to regulate blood sugar."],
                "lifestyle": ["Engage in light exercise like walking daily.", "Practice stress-relieving activities like meditation."],
                "herbal_remedies": [
                    "Karela to regulate blood sugar levels.",
                    "Moringa for overall metabolic health.",
                    "Gokshura for improving vitality and maintaining kidney health."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "light",
            "texture": "normal",
            "dark_circles": "present",
            "spots": [{"location": "x=20, y=160", "size": 3}]
        },
        "nail_analysis": {
            "color": {"RGB": [210, 190, 170], "Hex": "#d2bea8"},
            "texture": {"edge_intensity": 0.3},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "normal",
            "thickness": "thin",
            "surface_changes": ["peeling"]
        },
        "outputs": {
            "disease": "Iron Deficiency",
            "dosha_analysis": {
                "vata": "mildly imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Peeling nails suggest nutrient deficiency.",
                "Dark circles indicate low hemoglobin levels."
            ],
            "recommendations": {
                "diet": ["Consume iron-rich foods like spinach and lentils.", "Pair with vitamin C-rich foods for better absorption."],
                "lifestyle": ["Avoid caffeine immediately after meals to improve iron absorption.", "Practice regular breathing exercises to improve oxygenation."],
                "herbal_remedies": [
                    "Arjuna for improving circulation.",
                    "Ashoka for replenishing blood and balancing Vata.",
                    "Khadira for purifying the blood."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "light",
            "texture": "smooth",
            "dark_circles": "present",
            "spots": [{"location": "x=25, y=150", "size": 2}]
        },
        "nail_analysis": {
            "color": {"RGB": [210, 190, 160], "Hex": "#d2bea0"},
            "texture": {"edge_intensity": 0.3},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "normal",
            "thickness": "thin",
            "surface_changes": ["peeling"]
        },
        "outputs": {
            "disease": "Vitamin C Deficiency (Scurvy)",
            "dosha_analysis": {
                "vata": "imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Peeling nails suggest a vitamin C deficiency.",
                "Smooth skin indicates no significant inflammation."
            ],
            "recommendations": {
                "diet": ["Consume citrus fruits like oranges and lemons.", "Include amla and pomegranate juice in daily meals."],
                "lifestyle": ["Drink aloe vera juice for internal hydration.", "Avoid excessive processed foods that deplete nutrients."],
                "herbal_remedies": [
                    "Amla Powder for boosting vitamin C levels.",
                    "Neem Face Pack to improve skin health.",
                    "Cucumber for hydration and skin rejuvenation."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "medium",
            "texture": "oily",
            "dark_circles": "absent",
            "spots": [{"location": "x=30, y=180", "size": 4}]
        },
        "nail_analysis": {
            "color": {"RGB": [180, 150, 120], "Hex": "#b49678"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "normal",
            "thickness": "thick",
            "surface_changes": ["yellow discoloration"]
        },
        "outputs": {
            "disease": "Fungal Nail Infection",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "mildly imbalanced",
                "kapha": "severely imbalanced"
            },
            "observations": [
                "Thick, yellow nails indicate a fungal infection.",
                "Oily skin suggests Kapha dominance."
            ],
            "recommendations": {
                "diet": ["Avoid sugary and processed foods.", "Consume bitter melon juice for antifungal benefits."],
                "lifestyle": ["Keep nails clean and dry.", "Use breathable footwear to reduce moisture."],
                "herbal_remedies": [
                    "Neem Oil for topical antifungal application.",
                    "Vidanga Powder for systemic antifungal effects.",
                    "Fenugreek Water to support immunity."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "dark",
            "texture": "rough",
            "dark_circles": "present",
            "spots": [{"location": "x=40, y=200", "size": 6}]
        },
        "nail_analysis": {
            "color": {"RGB": [90, 70, 50], "Hex": "#5a4632"},
            "texture": {"edge_intensity": 0.5},
            "shape": {"shapes_detected": ["Square"]},
            "growth_rate": "slow",
            "thickness": "thin",
            "surface_changes": ["longitudinal ridges"]
        },
        "outputs": {
            "disease": "Rheumatoid Arthritis",
            "dosha_analysis": {
                "vata": "severely imbalanced",
                "pitta": "mildly imbalanced",
                "kapha": "balanced"
            },
            "observations": [
                "Thin nails with ridges suggest systemic inflammation.",
                "Rough skin reflects Vata dominance."
            ],
            "recommendations": {
                "diet": ["Consume warm, nourishing foods like ghee and soups.", "Include turmeric milk for its anti-inflammatory properties."],
                "lifestyle": ["Massage joints with sesame oil.", "Practice yoga to improve joint mobility."],
                "herbal_remedies": [
                    "Boswellia for joint inflammation.",
                    "Kushta for reducing pain and stiffness.",
                    "Ginger Tea to improve circulation and reduce inflammation."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "medium",
            "texture": "normal",
            "dark_circles": "absent",
            "spots": [{"location": "x=35, y=170", "size": 4}]
        },
        "nail_analysis": {
            "color": {"RGB": [170, 140, 120], "Hex": "#aa8c78"},
            "texture": {"edge_intensity": 0.3},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "normal",
            "thickness": "thin",
            "surface_changes": ["white spots"]
        },
        "outputs": {
            "disease": "Zinc Deficiency",
            "dosha_analysis": {
                "vata": "slightly imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "White spots on nails indicate zinc deficiency.",
                "Normal skin texture suggests no systemic inflammation."
            ],
            "recommendations": {
                "diet": ["Consume zinc-rich foods like pumpkin seeds and lentils.", "Include curry leaves in meals for added nutrients."],
                "lifestyle": ["Avoid nail trauma and harsh detergents.", "Massage nails with coconut oil for hydration."],
                "herbal_remedies": [
                    "Aloe Vera Juice for hydration.",
                    "Coriander to support detoxification.",
                    "Chia Seeds for improving nutrient absorption."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "light",
            "texture": "normal",
            "dark_circles": "present",
            "spots": [{"location": "x=20, y=150", "size": 2}]
        },
        "nail_analysis": {
            "color": {"RGB": [210, 180, 140], "Hex": "#d2b48c"},
            "texture": {"edge_intensity": 0.2},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "normal",
            "thickness": "thin",
            "surface_changes": ["peeling"]
        },
        "outputs": {
            "disease": "Vitamin D Deficiency",
            "dosha_analysis": {
                "vata": "imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Peeling nails suggest a lack of essential nutrients.",
                "Dark circles signal mild Vata imbalance."
            ],
            "recommendations": {
                "diet": ["Consume vitamin D-rich foods like fish and fortified milk.", "Include sesame oil in cooking for added calcium absorption."],
                "lifestyle": ["Expose skin to sunlight for 15 minutes daily.", "Practice calming yoga for stress relief."],
                "herbal_remedies": [
                    "Shatavari Syrup to boost bone health.",
                    "Moringa Powder for its calcium and vitamin D content.",
                    "Fenugreek Tea to improve nutrient absorption."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "dark",
            "texture": "dry",
            "dark_circles": "absent",
            "spots": [{"location": "x=40, y=180", "size": 5}]
        },
        "nail_analysis": {
            "color": {"RGB": [100, 80, 60], "Hex": "#64503c"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Square"]},
            "growth_rate": "slow",
            "thickness": "thin",
            "surface_changes": ["ridges"]
        },
        "outputs": {
            "disease": "Hypothyroidism",
            "dosha_analysis": {
                "vata": "imbalanced",
                "pitta": "balanced",
                "kapha": "severely imbalanced"
            },
            "observations": [
                "Slow nail growth and ridges suggest thyroid imbalance.",
                "Dry skin indicates Kapha dominance."
            ],
            "recommendations": {
                "diet": ["Include iodine-rich foods like seaweed.", "Consume ginger tea to improve metabolism."],
                "lifestyle": ["Practice yoga to stimulate the thyroid gland.", "Avoid processed and high-sugar foods."],
                "herbal_remedies": [
                    "Kanchanar for thyroid health.",
                    "Cinnamon Powder to support metabolism.",
                    "Jatamansi to reduce stress and balance hormones."
                ]
            }
        }
    },
     {
        "skin_analysis": {
            "tone": "medium",
            "texture": "oily",
            "dark_circles": "present",
            "spots": [{"location": "x=30, y=170", "size": 3}]
        },
        "nail_analysis": {
            "color": {"RGB": [170, 140, 120], "Hex": "#aa8c78"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "normal",
            "thickness": "thick",
            "surface_changes": ["yellow discoloration"]
        },
        "outputs": {
            "disease": "Fungal Nail Infection",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "mildly imbalanced",
                "kapha": "severely imbalanced"
            },
            "observations": [
                "Yellow discoloration on nails indicates fungal infection.",
                "Oily skin suggests Kapha imbalance."
            ],
            "recommendations": {
                "diet": ["Avoid sugar and processed foods.", "Include bitter melon and neem leaves."],
                "lifestyle": ["Keep nails dry and clean.", "Wear breathable footwear."],
                "herbal_remedies": [
                    "Neem oil for topical application.",
                    "Vidanga for anti-parasitic effects.",
                    "Dandelion tea for detoxification."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "light",
            "texture": "dry",
            "dark_circles": "absent",
            "spots": [{"location": "x=25, y=150", "size": 2}]
        },
        "nail_analysis": {
            "color": {"RGB": [210, 180, 140], "Hex": "#d2b48c"},
            "texture": {"edge_intensity": 0.3},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "slow",
            "thickness": "thin",
            "surface_changes": ["peeling"]
        },
        "outputs": {
            "disease": "Vitamin A Deficiency",
            "dosha_analysis": {
                "vata": "mildly imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Peeling nails suggest vitamin deficiency.",
                "Dry skin indicates reduced hydration and nutrient depletion."
            ],
            "recommendations": {
                "diet": ["Include carrots, spinach, and sweet potatoes for vitamin A.", "Consume ghee for better nutrient absorption."],
                "lifestyle": ["Avoid prolonged exposure to cold environments.", "Use aloe vera gel for skin hydration."],
                "herbal_remedies": [
                    "Amla powder for improving skin and hair health.",
                    "Turmeric face mask for skin rejuvenation.",
                    "Papaya juice for hydration and vitamin A."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "dark",
            "texture": "normal",
            "dark_circles": "present",
            "spots": [{"location": "x=40, y=190", "size": 4}]
        },
        "nail_analysis": {
            "color": {"RGB": [150, 130, 100], "Hex": "#968264"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Square"]},
            "growth_rate": "normal",
            "thickness": "thick",
            "surface_changes": ["white spots"]
        },
        "outputs": {
            "disease": "Zinc Deficiency",
            "dosha_analysis": {
                "vata": "slightly imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "White spots on nails indicate zinc deficiency.",
                "Dark circles suggest fatigue and mild nutrient depletion."
            ],
            "recommendations": {
                "diet": ["Consume pumpkin seeds, lentils, and chickpeas for zinc.", "Include fenugreek tea for improving metabolism."],
                "lifestyle": ["Avoid heavy physical exertion until deficiency is addressed.", "Use rose water to hydrate skin."],
                "herbal_remedies": [
                    "Coriander seeds for detoxification.",
                    "Fenugreek water to support nutrient absorption.",
                    "Moringa powder for overall nutrient replenishment."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "medium",
            "texture": "rough",
            "dark_circles": "absent",
            "spots": [{"location": "x=35, y=170", "size": 3}]
        },
        "nail_analysis": {
            "color": {"RGB": [180, 160, 130], "Hex": "#b49c82"},
            "texture": {"edge_intensity": 0.5},
            "shape": {"shapes_detected": ["Oval"]},
            "growth_rate": "slow",
            "thickness": "thin",
            "surface_changes": ["longitudinal ridges"]
        },
        "outputs": {
            "disease": "Hypothyroidism",
            "dosha_analysis": {
                "vata": "imbalanced",
                "pitta": "balanced",
                "kapha": "imbalanced"
            },
            "observations": [
                "Longitudinal ridges on nails suggest thyroid dysfunction.",
                "Rough skin indicates metabolic imbalance."
            ],
            "recommendations": {
                "diet": ["Include iodine-rich foods like seaweed and fish.", "Consume sesame oil for better thyroid support."],
                "lifestyle": ["Practice yoga to stimulate thyroid function.", "Avoid foods that interfere with thyroid health, such as soy."],
                "herbal_remedies": [
                    "Kanchanar powder for thyroid health.",
                    "Triphala for detoxification.",
                    "Shankhapushpi for calming the mind."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "dark",
            "texture": "oily",
            "dark_circles": "present",
            "spots": [{"location": "x=50, y=200", "size": 5}]
        },
        "nail_analysis": {
            "color": {"RGB": [120, 90, 70], "Hex": "#785a46"},
            "texture": {"edge_intensity": 0.4},
            "shape": {"shapes_detected": ["Square"]},
            "growth_rate": "normal",
            "thickness": "thick",
            "surface_changes": ["yellow discoloration"]
        },
        "outputs": {
            "disease": "Type 2 Diabetes",
            "dosha_analysis": {
                "vata": "balanced",
                "pitta": "mildly imbalanced",
                "kapha": "severely imbalanced"
            },
            "observations": [
                "Yellow nails and oily skin indicate Kapha imbalance and metabolic dysfunction.",
                "Dark circles suggest poor circulation and fatigue."
            ],
            "recommendations": {
                "diet": ["Avoid high-sugar and processed foods.", "Include karela and cinnamon powder for blood sugar regulation."],
                "lifestyle": ["Practice daily walks to improve circulation.", "Engage in regular stress-relieving activities like meditation."],
                "herbal_remedies": [
                    "Fenugreek tea for regulating metabolism.",
                    "Ginger tea for improving digestion.",
                    "Moringa powder for nutrient support."
                ]
            }
        }
    },
    {
        "skin_analysis": {
            "tone": "medium",
            "texture": "dry",
            "dark_circles": "absent",
            "spots": [{"location": "x=20, y=150", "size": 3}]
        },
        "nail_analysis": {
            "color": {"RGB": [190, 160, 130], "Hex": "#bea082"},
            "texture": {"edge_intensity": 0.3},
            "shape": {"shapes_detected": ["Round"]},
            "growth_rate": "slow",
            "thickness": "thin",
            "surface_changes": ["peeling"]
        },
        "outputs": {
            "disease": "Calcium Deficiency",
            "dosha_analysis": {
                "vata": "severely imbalanced",
                "pitta": "balanced",
                "kapha": "balanced"
            },
            "observations": [
                "Peeling nails and dry skin indicate calcium deficiency.",
                "Slow nail growth suggests systemic Vata imbalance."
            ],
            "recommendations": {
                "diet": ["Include calcium-rich foods like milk, almonds, and leafy greens.", "Consume sesame seeds for bone health."],
                "lifestyle": ["Avoid overexertion and prioritize restful sleep.", "Use warm mustard oil for skin hydration."],
                "herbal_remedies": [
                    "Ashwagandha for improving bone strength.",
                    "Musta for aiding digestion and nutrient absorption.",
                    "Licorice tea for reducing dryness."
                ]
            }
        }
    }
]
!(register-module! ../Neuro-Symbolic-Expert-System)
!(import! &self main)

(= (decisionMaker $ml)
    (if (== (get-metatype $ml) Grounded)
        (if (> $ml 55)
            ("No Need of Watering plant with " $ml "% moisture")
            ("Water your plant moisture level below threshold!! moisture level at: " $ml%)
        )
        ("Error: Must be of Type Number")
    )
)

!(decisionMaker (superpose moisture_prediction))
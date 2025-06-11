
(= (decisionMaker $ml)
    (if (== (get-metatype $ml) Grounded)
        (if (> $ml 55)
            ("No Need of Watering plant with " $ml "% moisture")
            ("Water your plant moisture level below threshold!! moisture level at: " $ml%)
        )
        ("Error: Must be of Type Number")
    )
)

!(decisionMaker (superpose (71.31 66.27 53.57)))
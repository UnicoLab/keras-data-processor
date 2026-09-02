# TextVectorizerOutputOptions

Output modes accepted by `TextFeature(output_mode=...)`.

The members are the exact strings `keras.layers.TextVectorization` and
KDP's own checks compare against. They used to be `auto()` integers here
while `kdp.processor` defined a second, string-valued class of the same
name, so whichever one a caller imported decided whether their option
worked or was quietly discarded.



# TextAccumulator

## Constructor

```python
__init__(self) -> None
```

Initializes the accumulator for text values, where each entry is a list of words separated by spaces.- **Attributes**: words (tf.Variable): TensorFlow variable to store unique words as strings.


---

## get_unique_words

```python
get_unique_words(self) -> list
```

Returns the unique words accumulated so far as a list of strings.

### Returns

    list of- **str**: Unique words accumulated.


---


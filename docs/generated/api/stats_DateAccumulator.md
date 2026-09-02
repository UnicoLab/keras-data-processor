# DateAccumulator

Accumulator for computing statistics of date features including cyclical encoding.

## Constructor

```python
__init__(self)
```

Initializes the accumulators for date features.

---

## mean

```python
mean(self) -> dict
```

Returns the mean statistics for date features.

---

## update

```python
update(self, dates: tensorflow.python.framework.tensor.Tensor) -> None
```

Updates the accumulators with new date values.

### Parameters- **dates**: Either a tensor of date strings (``YYYY-MM-DD`` or ``YYYY/MM/DD``),
        as read from a CSV column, or an already parsed numeric tensor of
        shape ``[batch_size, >=4]`` whose columns are
        ``[year, month, day_of_month, day_of_week]``.

### Raises
- **ValueError**: If a parsed numeric tensor does not carry the four
        expected date components.


---

## variance

```python
variance(self) -> dict
```

Returns the variance statistics for date features.

---


import tensorflow as tf
import numpy as np


Hi peoples, I calculated some possible scenarios for the future rent distribution:

————————
PAYMENTS
————————

RENT: 1982€
ELECTRICITY: 155€
INTERNET: 40€

Since we will most definitely distribute Internet and electricity equally, this would be 32.5€ per person each month.

—————
ROOMS
—————

Personal rooms:
Emlie: 2.9 m × 4.5 m = 13.05 m²
Anastasija: 3.0 m × 4.4 m = 13.20 m²
Mortimer: 3.2 m × 4.1 m = 13.12 m²
Sadaf: 2.9 m × 4.3 m = 12.47 m²
Hlib: 2.8 m × 4.4 m = 12.32 m²
Karla: 4.1 m × 4.1 m = 16.81 m²

Common rooms:
small WC: 1.5 m × 1.7 m
small shower: 2.0 m × 1.7 m
Entrance: 3.5 m × 3.2 m + 0.9 m × 0.6 m
living room: 3.2 m × 5.6 m + 1.6 m × 0.8 m
large itchen: 4.4 m × 2.1 m
small kitchen: 2.7 m × 2.1 m
large bath: 3.0 m × 2.0 m

Total:
Common living space: 57.8 m²
Total  living space: 138.8 m²

————————————————————————
OPTIONS (not exhaustive)
————————————————————————

Previous Payment
Emilie: 352€
Anastasija: 382€
Mortimer: 352€
Sadaf: 352€
Hlib: 363€
Karla: 372€

1. Rent proportional to PERSONAL + COMMON SPACE

Emilie: 328€/360.5€  delta 7.5€
Anastasija: 329€/361.5€ delta -20.5€
Mortimer: 329€/361.5€ delta 9.5€
Sadaf: 326€/358.5€ delta 6.5€
Hlib: 325€/357.5€ delta -6.5€
Karla: 346€/378.5€ delta 6.5€

2. Rent equally distributed
              
For everyone: 333.5€/363.5€

3. Interpolate
In this case we would choose a middle ground between option 1 and 2 by mixing them (for example) 50%/50% each, depending on how much we want to weight personal space versus common space.
Edited7:05 PM





VOTE
OPTION A Sadaf
Emilie		prev: 352,	new: 360.5,	diff: 8.5
Anastasija	prev: 382,	new: 361.5,	diff: -20.5
Mortimer	prev: 352,	new: 361.5,	diff: 9.5
Sadaf		prev: 352,	new: 358.5,	diff: 6.5
Hlib		prev: 363,	new: 357.5,	diff: -5.5
Karla		prev: 372,	new: 378.5,	diff: 6.5

VOTE B Karla, Emilie, Sadaf, Hlib
Emilie		prev: 352,	new: 361.25, diff: 9.25
Anastasija	prev: 382,	new: 362.0,	 diff: -20.0
Mortimer	prev: 352,	new: 362.0,	 diff: 10.0
Sadaf		prev: 352,	new: 359.75, diff: 7.75
Hlib		prev: 363,	new: 359.0,	 diff: -4.0
Karla		prev: 372,	new: 374.75, diff: 2.75

VOTE C Karla, Emilie
Emilie		prev: 352,	new: 362.0,	diff: 10.0
Anastasija	prev: 382,	new: 362.5,	diff: -19.5
Mortimer	prev: 352,	new: 362.5,	diff:  10.5
Sadaf		prev: 352,	new: 361.0,	diff:  9.0
Hlib		prev: 363,	new: 360.5,	diff: -2.5
Karla		prev: 372,	new: 371.0,	diff: -1.0



WINNER

Emilie		new: 361.25
Anastasija	new: 362.0
Mortimer	new: 362.0
Sadaf		new: 359.75
Hlib		new: 359.0
Karla		new: 374.75


Sadaf, Anastasija, Karla, Mortimer
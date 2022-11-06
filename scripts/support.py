"""Module containing a set of supporting classes and functions.

Author: Sivakumar Balasubramanian
Date: 27 Jun 2022
Email: siva82kb@gmail.com
"""

import datetime
import msgpack
import msgpack_numpy as msgpacknp

def encode_datetime(obj):
    if isinstance(obj, datetime.datetime):
        obj = {'__datetime__': True,
               'as_str': obj.strftime("%Y%m%dT%H:%M:%S.%f").encode()}
    return obj